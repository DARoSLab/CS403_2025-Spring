import mujoco
import numpy as np
import mujoco
from BoxControlHandler import BoxControlHandle

class YourCtrl:
  
  def __init__(self, m: mujoco.MjModel, d: mujoco.MjData):
    self.m = m
    self.d = d
    self.init_qpos = d.qpos.copy()

    self.boxCtrlhdl = BoxControlHandle(self.m,self.d)
    self.boxCtrlhdl.set_difficulty(0.25) #set difficulty level

   

  def update(self):

   for i in range(6):
       self.d.ctrl[i] = 150.0*(self.init_qpos[i] - self.d.qpos[i])  - 5.2 *self.d.qvel[i]
  
   

