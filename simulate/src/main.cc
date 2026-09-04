// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// !!! hack code: make glfw_adapter.window_ public
#define private public
#include "glfw_adapter.h"
#undef private

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <atomic>
#include <ctime>
#include <vector>

#include <mujoco/mujoco.h>
#include "simulate.h"
#include "array_safety.h"
#include "unitree_sdk2_bridge.h"
#include "param.h"

#define MUJOCO_PLUGIN_DIR "mujoco_plugin"

extern "C"
{
#if defined(_WIN32) || defined(__CYGWIN__)
#include <windows.h>
#else
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#include <sys/errno.h>
#include <unistd.h>
#endif
}

class ElasticBand
{
public:
  ElasticBand(){};
  void Advance(std::vector<double> x, std::vector<double> dx)
  {
    std::vector<double> delta_x = {0.0, 0.0, 0.0};
    delta_x[0] = point_[0] - x[0];
    delta_x[1] = point_[1] - x[1];
    delta_x[2] = point_[2] - x[2];
    double distance = sqrt(delta_x[0] * delta_x[0] + delta_x[1] * delta_x[1] + delta_x[2] * delta_x[2]);

    std::vector<double> direction = {0.0, 0.0, 0.0};
    direction[0] = delta_x[0] / distance;
    direction[1] = delta_x[1] / distance;
    direction[2] = delta_x[2] / distance;

    double v = dx[0] * direction[0] + dx[1] * direction[1] + dx[2] * direction[2];

    f_[0] = (stiffness_ * (distance - length_) - damping_ * v) * direction[0];
    f_[1] = (stiffness_ * (distance - length_) - damping_ * v) * direction[1];
    f_[2] = (stiffness_ * (distance - length_) - damping_ * v) * direction[2];
  }


  double stiffness_ = 200;
  double damping_ = 100;
  std::vector<double> point_ = {0, 0, 3};
  double length_ = 0.0;
  bool enable_ = true;
  std::vector<double> f_ = {0, 0, 0};
};
inline ElasticBand elastic_band;


namespace
{
  namespace mj = ::mujoco;
  namespace mju = ::mujoco::sample_util;

  // constants
  const double syncMisalign = 0.1;       // maximum mis-alignment before re-sync (simulation seconds)
  const double simRefreshFraction = 0.7; // fraction of refresh available for simulation
  const int kErrorLength = 1024;         // load error string length

  // model and data
  mjModel *m = nullptr;
  mjData *d = nullptr;

  // control noise variables
  mjtNum *ctrlnoise = nullptr;

  using Seconds = std::chrono::duration<double>;

  //---------------------------------------- plugin handling -----------------------------------------

  // return the path to the directory containing the current executable
  // used to determine the location of auto-loaded plugin libraries
  std::string getExecutableDir()
  {
#if defined(_WIN32) || defined(__CYGWIN__)
    constexpr char kPathSep = '\\';
    std::string realpath = [&]() -> std::string
    {
      std::unique_ptr<char[]> realpath(nullptr);
      DWORD buf_size = 128;
      bool success = false;
      while (!success)
      {
        realpath.reset(new (std::nothrow) char[buf_size]);
        if (!realpath)
        {
          std::cerr << "cannot allocate memory to store executable path\n";
          return "";
        }

        DWORD written = GetModuleFileNameA(nullptr, realpath.get(), buf_size);
        if (written < buf_size)
        {
          success = true;
        }
        else if (written == buf_size)
        {
          // realpath is too small, grow and retry
          buf_size *= 2;
        }
        else
        {
          std::cerr << "failed to retrieve executable path: " << GetLastError() << "\n";
          return "";
        }
      }
      return realpath.get();
    }();
#else
    constexpr char kPathSep = '/';
#if defined(__APPLE__)
    std::unique_ptr<char[]> buf(nullptr);
    {
      std::uint32_t buf_size = 0;
      _NSGetExecutablePath(nullptr, &buf_size);
      buf.reset(new char[buf_size]);
      if (!buf)
      {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }
      if (_NSGetExecutablePath(buf.get(), &buf_size))
      {
        std::cerr << "unexpected error from _NSGetExecutablePath\n";
      }
    }
    const char *path = buf.get();
#else
    const char *path = "/proc/self/exe";
#endif
    std::string realpath = [&]() -> std::string
    {
      std::unique_ptr<char[]> realpath(nullptr);
      std::uint32_t buf_size = 128;
      bool success = false;
      while (!success)
      {
        realpath.reset(new (std::nothrow) char[buf_size]);
        if (!realpath)
        {
          std::cerr << "cannot allocate memory to store executable path\n";
          return "";
        }

        std::size_t written = readlink(path, realpath.get(), buf_size);
        if (written < buf_size)
        {
          realpath.get()[written] = '\0';
          success = true;
        }
        else if (written == -1)
        {
          if (errno == EINVAL)
          {
            // path is already not a symlink, just use it
            return path;
          }

          std::cerr << "error while resolving executable path: " << strerror(errno) << '\n';
          return "";
        }
        else
        {
          // realpath is too small, grow and retry
          buf_size *= 2;
        }
      }
      return realpath.get();
    }();
#endif

    if (realpath.empty())
    {
      return "";
    }

    for (std::size_t i = realpath.size() - 1; i > 0; --i)
    {
      if (realpath.c_str()[i] == kPathSep)
      {
        return realpath.substr(0, i);
      }
    }

    // don't scan through the entire file system's root
    return "";
  }

  // scan for libraries in the plugin directory to load additional plugins
  void scanPluginLibraries()
  {
    // check and print plugins that are linked directly into the executable
    int nplugin = mjp_pluginCount();
    if (nplugin)
    {
      std::printf("Built-in plugins:\n");
      for (int i = 0; i < nplugin; ++i)
      {
        std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
      }
    }

    // define platform-specific strings
#if defined(_WIN32) || defined(__CYGWIN__)
    const std::string sep = "\\";
#else
    const std::string sep = "/";
#endif

    // try to open the ${EXECDIR}/plugin directory
    // ${EXECDIR} is the directory containing the simulate binary itself
    const std::string executable_dir = getExecutableDir();
    if (executable_dir.empty())
    {
      return;
    }

    const std::string plugin_dir = getExecutableDir() + sep + MUJOCO_PLUGIN_DIR;
    mj_loadAllPluginLibraries(
        plugin_dir.c_str(), +[](const char *filename, int first, int count)
                            {
        std::printf("Plugins registered by library '%s':\n", filename);
        for (int i = first; i < first + count; ++i) {
          std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
        } });
  }

  //------------------------------------------- simulation -------------------------------------------

  mjModel *LoadModel(const char *file, mj::Simulate &sim)
  {
    // this copy is needed so that the mju::strlen call below compiles
    char filename[mj::Simulate::kMaxFilenameLength];
    mju::strcpy_arr(filename, file);

    // make sure filename is not empty
    if (!filename[0])
    {
      return nullptr;
    }

    // load and compile
    char loadError[kErrorLength] = "";
    mjModel *mnew = 0;
    if (mju::strlen_arr(filename) > 4 &&
        !std::strncmp(filename + mju::strlen_arr(filename) - 4, ".mjb",
                      mju::sizeof_arr(filename) - mju::strlen_arr(filename) + 4))
    {
      mnew = mj_loadModel(filename, nullptr);
      if (!mnew)
      {
        mju::strcpy_arr(loadError, "could not load binary model");
      }
    }
    else
    {
      mnew = mj_loadXML(filename, nullptr, loadError, kErrorLength);
      // remove trailing newline character from loadError
      if (loadError[0])
      {
        int error_length = mju::strlen_arr(loadError);
        if (loadError[error_length - 1] == '\n')
        {
          loadError[error_length - 1] = '\0';
        }
      }
    }

    mju::strcpy_arr(sim.load_error, loadError);

    if (!mnew)
    {
      std::printf("%s\n", loadError);
      return nullptr;
    }

    // compiler warning: print and pause
    if (loadError[0])
    {
      // mj_forward() below will print the warning message
      std::printf("Model compiled, but simulation warning (paused):\n  %s\n", loadError);
      sim.run = 0;
    }

    return mnew;
  }

  // simulate in background thread (while rendering in main thread)
  void PhysicsLoop(mj::Simulate &sim)
  {
    // cpu-sim syncronization point
    std::chrono::time_point<mj::Simulate::Clock> syncCPU;
    mjtNum syncSim = 0;

    // ChannelFactory::Instance()->Init(0);
    // UnitreeDds ud(d);

    // run until asked to exit
    while (!sim.exitrequest.load())
    {
      if (sim.droploadrequest.load())
      {
        sim.LoadMessage(sim.dropfilename);
        mjModel *mnew = LoadModel(sim.dropfilename, sim);
        sim.droploadrequest.store(false);

        mjData *dnew = nullptr;
        if (mnew)
          dnew = mj_makeData(mnew);
        if (dnew)
        {
          sim.Load(mnew, dnew, sim.dropfilename);

          mj_deleteData(d);
          mj_deleteModel(m);

          m = mnew;
          d = dnew;
          mj_forward(m, d);

          // allocate ctrlnoise
          free(ctrlnoise);
          ctrlnoise = (mjtNum *)malloc(sizeof(mjtNum) * m->nu);
          mju_zero(ctrlnoise, m->nu);
        }
        else
        {
          sim.LoadMessageClear();
        }
      }

      if (sim.uiloadrequest.load())
      {
        sim.uiloadrequest.fetch_sub(1);
        sim.LoadMessage(sim.filename);
        mjModel *mnew = LoadModel(sim.filename, sim);
        mjData *dnew = nullptr;
        if (mnew)
          dnew = mj_makeData(mnew);
        if (dnew)
        {
          sim.Load(mnew, dnew, sim.filename);

          mj_deleteData(d);
          mj_deleteModel(m);

          m = mnew;
          d = dnew;
          mj_forward(m, d);

          // allocate ctrlnoise
          free(ctrlnoise);
          ctrlnoise = static_cast<mjtNum *>(malloc(sizeof(mjtNum) * m->nu));
          mju_zero(ctrlnoise, m->nu);
        }
        else
        {
          sim.LoadMessageClear();
        }
      }

      // sleep for 1 ms or yield, to let main thread run
      //  yield results in busy wait - which has better timing but kills battery life
      if (sim.run && sim.busywait)
      {
        std::this_thread::yield();
      }
      else
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }

      {
        // lock the sim mutex
        const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

        // run only if model is present
        if (m)
        {
          // running
          if (sim.run)
          {
            bool stepped = false;

            // record cpu time at start of iteration
            const auto startCPU = mj::Simulate::Clock::now();

            // elapsed CPU and simulation time since last sync
            const auto elapsedCPU = startCPU - syncCPU;
            double elapsedSim = d->time - syncSim;

            // inject noise
            if (sim.ctrl_noise_std)
            {
              // convert rate and scale to discrete time (Ornstein–Uhlenbeck)
              mjtNum rate = mju_exp(-m->opt.timestep / mju_max(sim.ctrl_noise_rate, mjMINVAL));
              mjtNum scale = sim.ctrl_noise_std * mju_sqrt(1 - rate * rate);

              for (int i = 0; i < m->nu; i++)
              {
                // update noise
                ctrlnoise[i] = rate * ctrlnoise[i] + scale * mju_standardNormal(nullptr);

                // apply noise
                d->ctrl[i] = ctrlnoise[i];
              }
            }

            // requested slow-down factor
            double slowdown = 100 / sim.percentRealTime[sim.real_time_index];

            // misalignment condition: distance from target sim time is bigger than syncmisalign
            bool misaligned =
                mju_abs(Seconds(elapsedCPU).count() / slowdown - elapsedSim) > syncMisalign;

            // out-of-sync (for any reason): reset sync times, step
            if (elapsedSim < 0 || elapsedCPU.count() < 0 || syncCPU.time_since_epoch().count() == 0 ||
                misaligned || sim.speed_changed)
            {
              // re-sync
              syncCPU = startCPU;
              syncSim = d->time;
              sim.speed_changed = false;

              // run single step, let next iteration deal with timing
              mj_step(m, d);
              stepped = true;
            }

            // in-sync: step until ahead of cpu
            else
            {
              bool measured = false;
              mjtNum prevSim = d->time;

              double refreshTime = simRefreshFraction / sim.refresh_rate;

              // step while sim lags behind cpu and within refreshTime
              while (Seconds((d->time - syncSim) * slowdown) < mj::Simulate::Clock::now() - syncCPU &&
                     mj::Simulate::Clock::now() - startCPU < Seconds(refreshTime))
              {
                // measure slowdown before first step
                if (!measured && elapsedSim)
                {
                  sim.measured_slowdown =
                      std::chrono::duration<double>(elapsedCPU).count() / elapsedSim;
                  measured = true;
                }

                // elastic band on base link
                if (param::config.enable_elastic_band == 1)
                {
                  if (elastic_band.enable_)
                  {
                    std::vector<double> x = {d->qpos[0], d->qpos[1], d->qpos[2]};
                    std::vector<double> dx = {d->qvel[0], d->qvel[1], d->qvel[2]};

                    elastic_band.Advance(x, dx);

                    d->xfrc_applied[param::config.band_attached_link] = elastic_band.f_[0];
                    d->xfrc_applied[param::config.band_attached_link + 1] = elastic_band.f_[1];
                    d->xfrc_applied[param::config.band_attached_link + 2] = elastic_band.f_[2];
                  }
                }

                // call mj_step
                mj_step(m, d);
                stepped = true;

                // break if reset
                if (d->time < prevSim)
                {
                  break;
                }
              }
            }

            // save current state to history buffer
            if (stepped)
            {
              sim.AddToHistory();
            }
          }

          // paused
          else
          {
            // run mj_forward, to update rendering and joint sliders
            mj_forward(m, d);
            sim.speed_changed = true;
          }
        }
      } // release std::lock_guard<std::mutex>
    }
  }
} // namespace

//-------------------------------------- physics_thread --------------------------------------------

void PhysicsThread(mj::Simulate *sim, const char *filename)
{
  // request loadmodel if file given (otherwise drag-and-drop)
  if (filename != nullptr)
  {
    sim->LoadMessage(filename);
    m = LoadModel(filename, *sim);
    if (m)
      d = mj_makeData(m);
    if (d)
    {
      sim->Load(m, d, filename);
      mj_forward(m, d);
      // LOCAL PATCH (cpp_control): sensordata is real from here on -- the
      // bridge may publish. See param::physics_ready.
      param::physics_ready = true;

      // allocate ctrlnoise
      free(ctrlnoise);
      ctrlnoise = static_cast<mjtNum *>(malloc(sizeof(mjtNum) * m->nu));
      mju_zero(ctrlnoise, m->nu);
    }
    else
    {
      sim->LoadMessageClear();
    }
  }

  PhysicsLoop(*sim);

  // delete everything we allocated
  free(ctrlnoise);
  mj_deleteData(d);
  mj_deleteModel(m);

  exit(0);
}

void *UnitreeSdk2BridgeThread(void *arg)
{
  // Wait for mujoco data.
  //
  // LOCAL PATCH (cpp_control): 500000 -> 1000 us.
  //
  // The physics thread sets `d` and immediately starts stepping, so every
  // microsecond spent here is the robot integrating with ctrl all-zero -- no
  // controller can have commanded anything yet, because the DDS bridge below
  // this loop is what publishes the state it would react to. At a 500 ms poll
  // that window measured 716 ms of simulated time on a G1, and a humanoid does
  // not survive it: cpp_control's difftrack sim2sim finds a sharp cliff at
  // ~350 ms, below which the standing hold recovers with zero torque
  // saturation and above which it is on the floor before the first command
  // lands. 1 ms brings the window to ~200 ms.
  //
  // Nothing else changes: same order, same condition, same everything after.
  while (true)
  {
    if (d)
    {
      std::cout << "Mujoco data is prepared" << std::endl;
      break;
    }
    usleep(1000);
  }

  // LOCAL PATCH (cpp_control): moved to main(), before the physics thread is
  // started. Creating the DDS participant needs neither `m` nor `d`, and doing
  // it here put it INSIDE the window in which the robot is already being
  // integrated with no controller. See the note on the poll above.


  int body_id = mj_name2id(m, mjOBJ_BODY, "torso_link");
  if (body_id < 0) {
    body_id = mj_name2id(m, mjOBJ_BODY, "base_link");
  }
  param::config.band_attached_link = 6 * body_id;

  std::unique_ptr<UnitreeSDK2BridgeBase> interface = nullptr;
  int idl_type = param::IDL_AUTO;
  try {
    idl_type = param::ResolveIdlType(param::config.robot, m->nu, param::config.idl_type);
  } catch (const std::invalid_argument& error) {
    std::cerr << error.what() << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (idl_type == param::IDL_HG) {
    std::cout << "Using unitree_hg DDS IDL for robot " << param::config.robot << std::endl;
    interface = std::make_unique<G1Bridge>(m, d);
  } else {
    std::cout << "Using unitree_go DDS IDL for robot " << param::config.robot << std::endl;
    interface = std::make_unique<Go2Bridge>(m, d);
  }
  interface->start();
  
  while (true)
  {
    sleep(1);
  }
}
//------------------------------------------ main --------------------------------------------------

// machinery for replacing command line error by a macOS dialog box when running under Rosetta
#if defined(__APPLE__) && defined(__AVX__)
extern void DisplayErrorDialogBox(const char *title, const char *msg);
static const char *rosetta_error_msg = nullptr;
__attribute__((used, visibility("default"))) extern "C" void _mj_rosettaError(const char *msg)
{
  rosetta_error_msg = msg;
}
#endif

// LOCAL PATCH (cpp_control): F9 video recording.
//
// unitree_mujoco's viewer is MuJoCo's stock `simulate`, which has no recorder --
// the drcl plant's equivalent lives in cpp_control/scripts/frame_recorder.py,
// and went with that plant. This is the same design in C++, and the same
// reasoning: the scene is rendered a SECOND time into an offscreen buffer and
// the raw RGB is piped to ffmpeg, rather than screen-grabbing the window.
// Offscreen because the video resolution is then independent of the window
// size, the capture is unaffected by occlusion or minimisation, and it does not
// care what the compositor reports the drawable size to be (this rig runs
// XWayland, where that goes wrong).
//
// WHAT IS IN THE VIDEO: the scene as the viewer is showing it. The camera and
// the visualisation options are read from the live `sim->cam` / `sim->opt`
// every frame, so the follow camera, an orbit with the mouse and a toggled
// visualisation all show up. MuJoCo's on-screen UI panels are drawn straight to
// the window and never reach here.
//
// THE CLOCK: frames are emitted on a schedule driven by SIMULATION time and the
// last render is duplicated to fill a gap, so the file is a constant-rate
// stream at `fps` whatever the loop does -- it plays back at 1x and pauses are
// cut out. DRCL_RECORD_CLOCK=wall records what you watched instead.
//
// THREADING: the key callback runs on the UI thread, which owns the window's GL
// context; this thread owns its own (a hidden window created on the main
// thread, because GLFW requires window creation there). So F9 only sets an
// atomic and every GL call stays on one thread. `mjv_updateScene` needs `m` and
// `d` alive, so it runs under sim->mtx; the render and the readback do not, and
// stay outside it.
namespace rec {

std::atomic_bool toggle_request{false};   // set by the UI thread on F9
std::atomic_bool is_recording{false};

std::string Env(const char* primary, const char* fallback, const char* dflt) {
  if (const char* v = std::getenv(primary)) { if (*v) return v; }
  if (const char* v = std::getenv(fallback)) { if (*v) return v; }
  return dflt;
}

int EnvInt(const char* primary, const char* fallback, int dflt) {
  try {
    return std::stoi(Env(primary, fallback, std::to_string(dflt).c_str()));
  } catch (...) {
    return dflt;
  }
}

std::string Stamp() {
  std::time_t t = std::time(nullptr);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&t));
  return std::string(buf);
}

void CaptureThread(mj::Simulate* sim, GLFWwindow* window) {
  glfwMakeContextCurrent(window);

  // Settings, read once. Same names frame_recorder.py reads, so a shell set up
  // for the mj_sim recorder drives this one unchanged.
  const std::string dir = Env("DRCL_RECORD_DIR", "CRL_RECORD_DIR", "recordings");
  const std::string size = Env("DRCL_RECORD_SIZE", "CRL_RECORD_SIZE", "1280x720");
  const std::string encoder = Env("DRCL_RECORD_ENCODER", "CRL_RECORD_ENCODER", "libx264");
  const std::string ffmpeg = Env("DRCL_RECORD_FFMPEG", "CRL_RECORD_FFMPEG", "ffmpeg");
  const std::string prefix = Env("DRCL_RECORD_PREFIX", "CRL_RECORD_PREFIX", "unitree_mujoco");
  const bool sim_clock = Env("DRCL_RECORD_CLOCK", "CRL_RECORD_CLOCK", "sim") != std::string("wall");
  const int fps = EnvInt("DRCL_RECORD_FPS", "CRL_RECORD_FPS", 30);
  const int crf = EnvInt("DRCL_RECORD_CRF", "CRL_RECORD_CRF", 23);
  int W = 1280, H = 720;
  if (std::sscanf(size.c_str(), "%dx%d", &W, &H) != 2) { W = 1280; H = 720; }

  mjvScene scn;
  mjv_defaultScene(&scn);
  mjrContext con;
  mjr_defaultContext(&con);
  const mjModel* built_for = nullptr;

  // DRCL_RECORD_AUTOSTART=1 is F9 pressed for you on the first loaded model:
  // it is what an unattended run needs (run_difftrack_sim2sim.sh -R), and it
  // means a recording can be taken on a machine with no keyboard on the window.
  bool autostart = Env("DRCL_RECORD_AUTOSTART", "CRL_RECORD_AUTOSTART", "0") == std::string("1");

  std::vector<unsigned char> rgb;
  FILE* pipe = nullptr;
  std::string path;
  double next_frame = 0.0;
  const double frame_dt = 1.0 / fps;
  auto wall0 = std::chrono::steady_clock::now();

  auto stop = [&]() {
    if (pipe) {
      pclose(pipe);
      pipe = nullptr;
      std::printf("[record] stopped: %s\n", path.c_str());
      std::fflush(stdout);
    }
    is_recording.store(false);
  };

  while (!sim->exitrequest.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));

    // ---- act on F9 -----------------------------------------------------
    if (toggle_request.exchange(false)) {
      if (pipe) {
        stop();
      } else if (built_for) {
        std::filesystem::create_directories(dir);
        path = dir + "/" + prefix + "_" + Stamp() + ".mp4";
        // -vf vflip: mjr_readPixels hands back rows bottom-up.
        char cmd[2048];
        std::snprintf(cmd, sizeof(cmd),
                      "%s -y -loglevel error -f rawvideo -pix_fmt rgb24 "
                      "-s %dx%d -r %d -i - -vf vflip -an -c:v %s -preset fast "
                      "-crf %d -pix_fmt yuv420p '%s'",
                      ffmpeg.c_str(), W, H, fps, encoder.c_str(), crf, path.c_str());
        pipe = popen(cmd, "w");
        if (!pipe) {
          std::printf("[record] could not start '%s'\n", ffmpeg.c_str());
        } else {
          is_recording.store(true);
          next_frame = 0.0;   // seeded from the first frame below
          wall0 = std::chrono::steady_clock::now();
          std::printf("[record] recording %dx%d@%d to %s\n", W, H, fps, path.c_str());
        }
        std::fflush(stdout);
      } else {
        std::printf("[record] no model loaded yet\n");
        std::fflush(stdout);
      }
    }

    // ---- snapshot the scene (needs m and d alive) -----------------------
    double now = 0.0;
    {
      const mj::MutexLock lock(sim->mtx);
      if (!m || !d) continue;
      if (built_for != m) {
        if (built_for) {
          mjv_freeScene(&scn);
          mjr_freeContext(&con);
        }
        mjv_defaultScene(&scn);
        mjv_makeScene(m, &scn, 2000);
        mjr_defaultContext(&con);
        mjr_makeContext(m, &con, mjFONTSCALE_100);
        built_for = m;
        // The offscreen buffer is a MODEL property (visual/global offwidth,
        // offheight, default 640x480). A viewport bigger than it is silently
        // clipped, so clamp and say why rather than write a cropped video.
        if (W > con.offWidth || H > con.offHeight) {
          std::printf("[record] scene offscreen buffer is %dx%d; clamping the "
                      "recording from %dx%d. Raise <visual><global offwidth= "
                      "offheight=> in the scene for more.\n",
                      con.offWidth, con.offHeight, W, H);
          W = std::min(W, con.offWidth);
          H = std::min(H, con.offHeight);
        }
        rgb.assign(static_cast<size_t>(3) * W * H, 0);
        if (autostart) {
          autostart = false;
          toggle_request.store(true);
        }
      }
      if (!pipe) continue;
      mjv_updateScene(m, d, &sim->opt, nullptr, &sim->cam, mjCAT_ALL, &scn);
      now = sim_clock ? d->time
                      : std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - wall0).count();
    }

    // ---- how many frames does this render owe? -------------------------
    if (next_frame == 0.0) next_frame = now;
    int due = 0;
    while (next_frame <= now && due < 4 * fps) {
      due++;
      next_frame += frame_dt;
    }
    if (due == 0) continue;

    // ---- render offscreen and write ------------------------------------
    const mjrRect viewport = {0, 0, W, H};
    mjr_setBuffer(mjFB_OFFSCREEN, &con);
    mjr_render(viewport, &scn, &con);
    mjr_readPixels(rgb.data(), nullptr, viewport, &con);
    for (int i = 0; i < due && pipe; i++) {
      if (std::fwrite(rgb.data(), 1, rgb.size(), pipe) != rgb.size()) {
        std::printf("[record] ffmpeg went away; stopping\n");
        stop();
      }
    }
  }

  stop();
  if (built_for) {
    mjv_freeScene(&scn);
    mjr_freeContext(&con);
  }
}

}  // namespace rec

// LOCAL PATCH (cpp_control): the key callback MuJoCo installed, so its own keys
// keep working. glfwSetKeyCallback allows exactly ONE callback per window, and
// the line at the bottom of main() replaced GlfwAdapter's -- which is the whole
// of MuJoCo's keyboard UI. Everything documented for `simulate` (space to
// pause, `[` / `]` to cycle cameras, Esc for the free camera, F1 help, Ctrl+P
// screenshot, backspace reset) has therefore been dead in unitree_mujoco, and
// silently: keys that do nothing look like keys you are pressing wrong.
// Chaining restores all of it and costs nothing.
static GLFWkeyfun prev_key_cb = nullptr;

// user keyboard callback
void user_key_cb(GLFWwindow* window, int key, int scancode, int act, int mods) {
  if (act==GLFW_PRESS)
  {
    if(param::config.enable_elastic_band == 1) {
      if (key==GLFW_KEY_9) {
        elastic_band.enable_ = !elastic_band.enable_;
      } else if (key==GLFW_KEY_7 || key==GLFW_KEY_UP) {
        elastic_band.length_ -= 0.1;
      } else if (key==GLFW_KEY_8 || key==GLFW_KEY_DOWN) {
        elastic_band.length_ += 0.1;
      }
    }
    if(key==GLFW_KEY_BACKSPACE) {
      mj_resetData(m, d);
      mj_forward(m, d);
    }
    // LOCAL PATCH (cpp_control): F9 starts/stops an mp4. Only an atomic is set
    // here -- this is the UI thread, and the recorder owns a GL context and an
    // ffmpeg pipe that belong to its own. F9 is free: MuJoCo's UI takes F1-F5.
    if(key==GLFW_KEY_F9) {
      rec::toggle_request.store(true);
    }
  }
  // LOCAL PATCH (cpp_control): hand every key to MuJoCo's own handler, which
  // this callback displaced. Unconditional and after ours, so a key we consume
  // still reaches it -- none of the keys above (7/8/9, backspace, F9) is one
  // simulate binds.
  if (prev_key_cb) {
    prev_key_cb(window, key, scancode, act, mods);
  }
}

// run event loop
int main(int argc, char **argv)
{

  // display an error if running on macOS under Rosetta 2
#if defined(__APPLE__) && defined(__AVX__)
  if (rosetta_error_msg)
  {
    DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
    std::exit(1);
  }
#endif

  // print version, check compatibility
  std::printf("MuJoCo version %s\n", mj_versionString());
  if (mjVERSION_HEADER != mj_version())
  {
    mju_error("Headers and library have different versions");
  }

  // scan for libraries in the plugin directory to load additional plugins
  scanPluginLibraries();

  mjvCamera cam;
  mjv_defaultCamera(&cam);

  mjvOption opt;
  mjv_defaultOption(&opt);

  mjvPerturb pert;
  mjv_defaultPerturb(&pert);

  // Load simulation configuration
  std::filesystem::path proj_dir = std::filesystem::path(getExecutableDir()).parent_path();
  param::config.load_from_yaml(proj_dir / "config.yaml");
  param::helper(argc, argv);
  if(param::config.robot_scene.is_relative()) {
    param::config.robot_scene = proj_dir.parent_path() / "unitree_robots" / param::config.robot / param::config.robot_scene;
  }

  // simulate object encapsulates the UI
  auto sim = std::make_unique<mj::Simulate>(
    std::make_unique<mj::GlfwAdapter>(),
    &cam, &opt, &pert, /* is_passive = */ false);

  // LOCAL PATCH (cpp_control): DDS up before ANY physics runs.
  //
  // Everything between the first physics step and the first LowCmd is the robot
  // integrating limp, and it is not a small window -- the difftrack sim2sim
  // measures a hard cliff at ~350 ms, past which a standing G1 is on the floor
  // before a controller can touch it. Participant creation is the expensive
  // half and depends on nothing MuJoCo owns, so it belongs here rather than in
  // the bridge thread. Combined with the 1 ms poll above this takes the window
  // from 716 ms to well inside the budget.
  unitree::robot::ChannelFactory::Instance()->Init(param::config.domain_id,
                                                  param::config.interface);

  std::thread unitree_thread(UnitreeSdk2BridgeThread, nullptr);

  // LOCAL PATCH (cpp_control): --wait-for-cmd.
  //
  // Shrinking the limp window is not the same as closing it, and for a humanoid
  // the difference matters: measured here, a G1 standing in its reset pose has
  // its hips 0.33 rad out of place after a third of a second of zero torque,
  // and a controller handed THAT cannot recover it however good its gains are.
  // So when asked, do not integrate at all until somebody is commanding. The UI
  // comes up, the robot is drawn at its reset pose, and the clock starts on the
  // first LowCmd -- which is what makes a sim2sim number a measurement of a
  // policy rather than of a startup race.
  //
  // Opt-in. Without the flag nothing changes: a simulator started on its own,
  // or driven from a joystick, runs exactly as before.
  if (param::config.wait_for_cmd)
  {
    sim->run = 0;
    std::thread([&sim]() {
      while (!param::lowcmd_received.load())
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      std::cout << "First LowCmd received; starting physics" << std::endl;
      sim->run = 1;
    }).detach();
  }

  // start physics thread
  std::thread physicsthreadhandle(&PhysicsThread, sim.get(), param::config.robot_scene.c_str());

  // LOCAL PATCH (cpp_control): F9 recording.
  //
  // The recorder's GL context has to be a window, and GLFW requires windows to
  // be created on the main thread -- so it is made here, hidden, and handed to
  // the capture thread, which is where it is made current. Not shared with the
  // visible window's context: the two are used concurrently from two threads,
  // and an unshared context is the arrangement that makes that safe.
  GLFWwindow* main_window =
      static_cast<mj::GlfwAdapter*>(sim->platform_ui.get())->window_;
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  GLFWwindow* rec_window = glfwCreateWindow(64, 64, "recorder", nullptr, nullptr);
  glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
  std::thread recorderthreadhandle;
  if (rec_window) {
    recorderthreadhandle = std::thread(&rec::CaptureThread, sim.get(), rec_window);
    std::printf("F9: start/stop video recording\n");
  } else {
    std::printf("could not create the recorder window; F9 recording disabled\n");
  }

  // start simulation UI loop (blocking call)
  prev_key_cb = glfwSetKeyCallback(main_window, user_key_cb);
  sim->RenderLoop();
  if (recorderthreadhandle.joinable()) {
    recorderthreadhandle.join();
  }
  physicsthreadhandle.join();

  pthread_exit(NULL);
  return 0;
}
