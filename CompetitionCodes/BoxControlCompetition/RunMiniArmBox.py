import abc
import atexit
import contextlib
import math
import os
import queue
import sys
import threading
import time
from typing import Callable, Optional, Tuple, Union
import weakref
# import BoxControl 
import YourControlCode
from BoxControlHandler import BoxControlHandle

import glfw
import mujoco
from mujoco import _simulate
import numpy as np

if not glfw._glfw:  # pylint: disable=protected-access
  raise RuntimeError('GLFW dynamic library handle is not available')
else:
  _simulate.set_glfw_dlhandle(glfw._glfw._handle)  # pylint: disable=protected-access

# Logarithmically spaced realtime slow-down coefficients (percent).
PERCENT_REALTIME = (
    100, 80, 66, 50, 40, 33, 25, 20, 16, 13,
    10, 8, 6.6, 5, 4, 3.3, 2.5, 2, 1.6, 1.3,
    1, 0.8, 0.66, 0.5, 0.4, 0.33, 0.25, 0.2, 0.16, 0.13,
    0.1
)

# Maximum time mis-alignment before re-sync.
MAX_SYNC_MISALIGN = 0.1

# Fraction of refresh available for simulation.
SIM_REFRESH_FRACTION = 0.7

CallbackType = Callable[[mujoco.MjModel, mujoco.MjData], None]
LoaderType = Callable[[], Tuple[mujoco.MjModel, mujoco.MjData]]
KeyCallbackType = Callable[[int], None]

# Loader function that also returns a file path for the GUI to display.
_LoaderWithPathType = Callable[[], Tuple[mujoco.MjModel, mujoco.MjData, str]]
_InternalLoaderType = Union[LoaderType, _LoaderWithPathType]

_Simulate = _simulate.Simulate


class VibratingBox:
    def __init__(self, m, d, body_name):
        """
        Initializes the vibrating box.

        Parameters:
        m (MjModel): MuJoCo model.
        d (MjData): MuJoCo simulation data.
        body_name (str): Name of the body to vibrate.
        amplitude (float): Magnitude of the vibration displacement.
        frequency (float): Frequency of vibration in Hz.
        """
        self.m = m
        self.d = d
        self.body_name = body_name

        # Get body and joint IDs
        self.body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if self.body_id == -1:
            raise ValueError(f"Body '{body_name}' not found in MuJoCo model.")

        self.joint_id = m.jnt_qposadr[m.body_jntadr[self.body_id]]
        self.initial_x = d.qpos[6]      # Initial X position
        self.initial_y = d.qpos[7]  # Initial Y position


    def apply_vibration(self, curr_time, amplitude=0.15, frequency=0.3):
        """Applies vibration around the initial position."""

        # Compute sinusoidal displacement
        displacement_x = amplitude * np.sin(2 * np.pi * frequency * curr_time)
        displacement_y = amplitude * np.cos(2 * np.pi * frequency * curr_time)

        Kp = 750
        Kd = 30 
        pos_des = np.array([self.initial_x + displacement_x, self.initial_y + displacement_y])

        self.d.ctrl[6] = Kp * (pos_des[0] - self.d.qpos[6]) - Kd * self.d.qvel[6]
        self.d.ctrl[7] = Kp * (pos_des[1] - self.d.qpos[7]) - Kd * self.d.qvel[7]


class Handle:
  """A handle for interacting with a MuJoCo viewer."""

  def __init__(
      self,
      sim: _Simulate,
      cam: mujoco.MjvCamera,
      opt: mujoco.MjvOption,
      pert: mujoco.MjvPerturb,
      user_scn: Optional[mujoco.MjvScene],
  ):
    self._sim = weakref.ref(sim)
    self._cam = cam
    self._opt = opt
    self._pert = pert
    self._user_scn = user_scn

  @property
  def cam(self):
    return self._cam

  @property
  def opt(self):
    return self._opt

  @property
  def perturb(self):
    return self._pert

  @property
  def user_scn(self):
    return self._user_scn

  @property
  def m(self):
    sim = self._sim()
    if sim is not None:
      return sim.m
    return None

  @property
  def d(self):
    sim = self._sim()
    if sim is not None:
      return sim.d
    return None

  def close(self):
    sim = self._sim()
    if sim is not None:
      sim.exit()

  def _get_sim(self) -> Optional[_Simulate]:
    sim = self._sim()
    if sim is not None:
      try:
        return sim if sim.exitrequest == 0 else None
      except mujoco.UnexpectedError:
        # UnexpectedError is raised when accessing `exitrequest` after the
        # underlying simulate instance has been deleted in C++.
        return None
    return None

  def is_running(self) -> bool:
    return self._get_sim() is not None

  def lock(self):
    sim = self._get_sim()
    if sim is not None:
      return sim.lock()
    return contextlib.nullcontext()

  def sync(self):
    sim = self._get_sim()
    if sim is not None:
      sim.sync()  # locks internally

  def update_hfield(self, hfieldid: int):
    sim = self._get_sim()
    if sim is not None:
      sim.update_hfield(hfieldid)  # locks internally and blocks until done

  def update_mesh(self, meshid: int):
    sim = self._get_sim()
    if sim is not None:
      sim.update_mesh(meshid)  # locks internally and blocks until done

  def update_texture(self, texid: int):
    sim = self._get_sim()
    if sim is not None:
      sim.update_texture(texid)  # locks internally and blocks until done

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()


# Abstract base dispatcher class for systems that require UI calls to be made
# on a specific thread (e.g. macOS). This is subclassed by system-specific
# Python launcher (mjpython) to implement the required dispatching mechanism.
class _MjPythonBase(metaclass=abc.ABCMeta):

  def launch_on_ui_thread(
      self,
      model: mujoco.MjModel,
      data: mujoco.MjData,
      handle_return: Optional['queue.Queue[Handle]'],
      key_callback: Optional[KeyCallbackType],
  ):
    pass


# When running under mjpython, the launcher initializes this object.
_MJPYTHON: Optional[_MjPythonBase] = None


def _file_loader(path: str) -> _LoaderWithPathType:
  """Loads an MJCF model from file path."""

  def load(path=path) -> Tuple[mujoco.MjModel, mujoco.MjData, str]:
    if len(path) >= 4 and path[-4:] == '.mjb':
      m = mujoco.MjModel.from_binary_path(path)
    else:
      m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    return m, d, path

  return load


def _reload(
    simulate: _Simulate, loader: _InternalLoaderType,
    notify_loaded: Optional[Callable[[], None]] = None
) -> Optional[Tuple[mujoco.MjModel, mujoco.MjData]]:
  """Internal function for reloading a model in the viewer."""
  try:
    simulate.load_message('') # path is unknown at this point
    load_tuple = loader()
  except Exception as e:  # pylint: disable=broad-except
    simulate.load_error = str(e)
    simulate.load_message_clear()
  else:
    m, d = load_tuple[:2]

    # If the loader does not raise an exception then we assume that it
    # successfully created mjModel and mjData. This is specified in the type
    # annotation, but we perform a runtime assertion here as well to prevent
    # possible segmentation faults.
    assert m is not None and d is not None

    path = load_tuple[2] if len(load_tuple) == 3 else ''
    simulate.load(m, d, path)

    # Make sure any load_error message is cleared
    simulate.load_error = ''

    if notify_loaded:
      notify_loaded()

    return m, d

def _physics_loop(simulate: _Simulate, loader: Optional[_InternalLoaderType]):
  """Physics loop for the GUI, to be run in a separate thread."""
  m: mujoco.MjModel = None
  d: mujoco.MjData = None
  reload = True

  # CPU-sim synchronization point.
  synccpu = 0.0
  syncsim = 0.0
  vibrating_box = None  # Vibrating box controller
  diff_settings = None #difficulty settings
  box_handler = None #box env handler
  completed = False

  # Run until asked to exit.
  while not simulate.exitrequest:
    
    if simulate.droploadrequest:
      simulate.droploadrequest = 0
      loader = _file_loader(simulate.dropfilename)
      reload = True

    if simulate.uiloadrequest:
      simulate.uiloadrequest_decrement()
      reload = True

    if reload and loader is not None:
      result = _reload(simulate, loader)
      if result is not None:
        m, d = result
        # ee_ctrl = BoxControl.BoxCtrl(m,d)
        ee_ctrl = YourControlCode.YourCtrl(m,d)
        box_handler = ee_ctrl.boxCtrlhdl
        vibrating_box = VibratingBox(m, d, "box_mould")
      

    reload = False

    # Sleep for 1 ms or yield, to let main thread run.
    if simulate.run != 0 and simulate.busywait != 0:
      time.sleep(0)
    else:
      time.sleep(0.001)

    with simulate.lock():
      if m is not None:
        assert d is not None
        if simulate.run:
          stepped = False
          # Record CPU time at start of iteration.
          startcpu = time.time()

          elapsedcpu = startcpu - synccpu
          elapsedsim = d.time - syncsim

          # Update control signal
          if(not(box_handler.check_goal_reached()) and not(completed) and elapsedsim < 30):
            ee_ctrl.update()
          else:
            completed = True
            if(box_handler.check_goal_reached()):
              box_handler.print_complete_time()
            else:
              print("Time's up! You did not complete the task. Elapsed time: %.2f seconds"%elapsedsim)

            return
            for i in range(6):
              box_handler.d.ctrl[i] = 0

          # Apply vibration to box_mould
          diff_settings = box_handler.get_diff_params()
          vibrating_box.apply_vibration(curr_time = d.time,amplitude=diff_settings[1], frequency=diff_settings[2])

          # Requested slow-down factor.
          slowdown = 100 / PERCENT_REALTIME[simulate.real_time_index]

          # Misalignment: distance from target sim time > MAX_SYNC_MISALIGN.
          misaligned = abs(elapsedcpu / slowdown -
                           elapsedsim) > MAX_SYNC_MISALIGN

          
          # Out-of-sync (for any reason): reset sync times, step.
          if (elapsedsim < 0 or elapsedcpu < 0 or synccpu == 0 or misaligned or
              simulate.speed_changed):
            # Re-sync.
            synccpu = startcpu
            syncsim = d.time
            simulate.speed_changed = False
            # Run single step, let next iteration deal with timing.
            mujoco.mj_step(m, d)
            stepped = True

          # In-sync: step until ahead of cpu.
          else:
            measured = False
            prevsim = d.time
            refreshtime = SIM_REFRESH_FRACTION / simulate.refresh_rate
            # Step while sim lags behind CPU and within refreshtime.
            while (((d.time - syncsim) * slowdown <
                    (time.time() - synccpu)) and
                   ((time.time() - startcpu) < refreshtime)):
              # Measure slowdown before first step.
              if not measured and elapsedsim:
                simulate.measured_slowdown = elapsedcpu / elapsedsim
                measured = True

              # Call mj_step.
              # print("ctrl: ", d.ctrl)
              mujoco.mj_step(m, d)
              stepped = True

              # Break if reset.
              if d.time < prevsim:
                break

          # save current state to history buffer
          if (stepped):
            simulate.add_to_history()

        else:  # simulate.run is False: GUI is paused.

          # Run mj_forward, to update rendering and joint sliders.
          mujoco.mj_forward(m, d)
          simulate.speed_changed = True

def _launch_internal(
    model: Optional[mujoco.MjModel] = None,
    data: Optional[mujoco.MjData] = None,
    *,
    run_physics_thread: bool,
    loader: Optional[_InternalLoaderType] = None,
    handle_return: Optional['queue.Queue[Handle]'] = None,
    key_callback: Optional[KeyCallbackType] = None,
    show_left_ui: bool = True,
    show_right_ui: bool = True,
) -> None:
  """Internal API, so that the public API has more readable type annotations."""
  if model is None and data is not None:
    raise ValueError('mjData is specified but mjModel is not')
  elif callable(model) and data is not None:
    raise ValueError(
        'mjData should not be specified when an mjModel loader is used')
  elif loader is not None and model is not None:
    raise ValueError('model and loader are both specified')
  elif run_physics_thread and handle_return is not None:
    raise ValueError('run_physics_thread and handle_return are both specified')

  if loader is None and model is not None:

    def _loader(m=model, d=data) -> Tuple[mujoco.MjModel, mujoco.MjData]:
      if d is None:
        d = mujoco.MjData(m)
      return m, d

    loader = _loader

  cam = mujoco.MjvCamera()
  opt = mujoco.MjvOption()
  pert = mujoco.MjvPerturb()
  if model and not run_physics_thread:
    user_scn = mujoco.MjvScene(model, _Simulate.MAX_GEOM)
  else:
    user_scn = None
  simulate = _Simulate(
      cam, opt, pert, user_scn, run_physics_thread, key_callback
  )

  simulate.ui0_enable = show_left_ui
  simulate.ui1_enable = show_right_ui

  # Initialize GLFW if not using mjpython.
  if _MJPYTHON is None:
    if not glfw.init():
      raise mujoco.FatalError('could not initialize GLFW')
    atexit.register(glfw.terminate)

  notify_loaded = None
  if handle_return:
    notify_loaded = lambda: handle_return.put_nowait(
        Handle(simulate, cam, opt, pert, user_scn)
    )

  if run_physics_thread:
    side_thread = threading.Thread(
        target=_physics_loop, args=(simulate, loader))
  else:
    side_thread = threading.Thread(
        target=_reload, args=(simulate, loader, notify_loaded))

  def make_exit(simulate):
    def exit_simulate():
      simulate.exit()
    return exit_simulate

  exit_simulate = make_exit(simulate)
  atexit.register(exit_simulate)

  side_thread.start()
  simulate.render_loop()
  atexit.unregister(exit_simulate)
  side_thread.join()
  simulate.destroy()


def launch(
    model: Optional[mujoco.MjModel] = None,
    data: Optional[mujoco.MjData] = None,
    *,
    loader: Optional[LoaderType] = None,
    show_left_ui: bool = True,
    show_right_ui: bool = True,
) -> None:
  """Launches the Simulate GUI."""
  _launch_internal(
      model,
      data,
      run_physics_thread=True,
      loader=loader,
      show_left_ui=show_left_ui,
      show_right_ui=show_right_ui,
  )


def launch_from_path(path: str) -> None:
  """Launches the Simulate GUI from file path."""
  _launch_internal(run_physics_thread=True, loader=_file_loader(path))


def launch_passive(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    key_callback: Optional[KeyCallbackType] = None,
    show_left_ui: bool = True,
    show_right_ui: bool = True,
) -> Handle:
  """Launches a passive Simulate GUI without blocking the running thread."""
  if not isinstance(model, mujoco.MjModel):
    raise ValueError(f'`model` is not a mujoco.MjModel: got {model!r}')
  if not isinstance(data, mujoco.MjData):
    raise ValueError(f'`data` is not a mujoco.MjData: got {data!r}')
  if key_callback is not None and not callable(key_callback):
    raise ValueError(
        f'`key_callback` is not callable: got {key_callback!r}')

  mujoco.mj_forward(model, data)
  handle_return = queue.Queue(1)

  if sys.platform != 'darwin':
    thread = threading.Thread(
        target=_launch_internal,
        args=(model, data),
        kwargs=dict(
            run_physics_thread=False,
            handle_return=handle_return,
            key_callback=key_callback,
            show_left_ui=show_left_ui,
            show_right_ui=show_right_ui,
        ),
    )
    thread.daemon = True
    thread.start()
  else:
    if not isinstance(_MJPYTHON, _MjPythonBase):
      raise RuntimeError(
          '`launch_passive` requires that the Python script be run under '
          '`mjpython` on macOS')
    _MJPYTHON.launch_on_ui_thread(
        model,
        data,
        handle_return,
        key_callback,
        show_left_ui,
        show_right_ui,
    )

  return handle_return.get()


if __name__ == '__main__':
  # pylint: disable=g-bad-import-order
  from absl import app  # pylint: disable=g-import-not-at-top
  dir_path = os.path.dirname(os.path.realpath(__file__))

  def main(argv) -> None:
    launch_from_path(dir_path + "/Robot/miniArmBox.xml")

  app.run(main)
