import mujoco
import numpy as np


class BoxControlHandle:
    def __init__(self,m,d):
        self.tolerance = 5e-3
        self.amp = 0.05
        self.freq = 0.5
        self.m = m
        self.d = d
        self.completed = False
        self.ee_id = mujoco.mj_name2id(m, 1, "EE_box")

    def _get_ee_position(self):
        return self.d.xpos[self.ee_id].copy()
  
    def _get_ee_orientation(self):
        return self.d.xquat[self.ee_id].copy()
    
    #set the difficulty (0,1]  
    def set_difficulty(self,d):
        self.tolerance = 5e-3
        self.amp = d*0.1
        self.freq = d*2
    
    
    def get_diff_params(self):
        return(self.tolerance,self.amp,self.freq)
    
    def print_diff_params(self):
        print(self.tolerance,self.amp,self.freq)

    def print_complete_time(self):
        if(not(self.completed)):
            print("You completed in %.2f seconds!"%self.d.time)
            self.completed = True

    def get_EE_pos_err(self):
        box_sensor1_idx = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, "mould_pos_sensor1")
        box_sensor2_idx = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, "mould_pos_sensor2")
        box_sensor3_idx = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, "mould_pos_sensor3")
        box_sensor4_idx = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, "mould_pos_sensor4") 
    
        boxmould_pos1 = self.d.sensordata[box_sensor1_idx*3:box_sensor1_idx*3+3]
        boxmould_pos2 = self.d.sensordata[box_sensor2_idx*3:box_sensor2_idx*3+3]
        boxmould_pos3 = self.d.sensordata[box_sensor3_idx*3:box_sensor3_idx*3+3]
        boxmould_pos4 = self.d.sensordata[box_sensor4_idx*3:box_sensor4_idx*3+3]

        box_ori,_ = self.box_orientation(boxmould_pos1, boxmould_pos2,boxmould_pos3,boxmould_pos4)
        target_ori = self.rotate_quat_90_y(box_ori)

        final_position = self.box_midpoint(boxmould_pos1, boxmould_pos2, boxmould_pos3, boxmould_pos4)
   
        target_ori_rtx = self.quat2SO3(target_ori)
        mid_target = final_position + target_ori_rtx @ np.array([-0.15,0,0.0])
        final_position += target_ori_rtx @ np.array([0.02,0,0.0])
    
        pre_time = 3
        if self.d.time < pre_time:
            target_position = mid_target 
        else:
            target_position = self.pos_interpolate(mid_target, final_position, self.d.time-pre_time, 5)
        EE_pos = self._get_ee_position()
        pos_err = (target_position - EE_pos)
        return pos_err

    def check_goal_reached(self):
      if self.ee_box_collision():
          box_body_id = mujoco.mj_name2id(self.m, 1, "box_mould")
          geom_ids = np.where(self.m.geom_bodyid == box_body_id)[0]
          green_color = np.array([0.0, 1.0, 0.0, 1.0])  # Green color
          for geom_id in geom_ids:
            self.m.geom_rgba[geom_id] = green_color
          self.d.ctrl[:6] = 0
          return True
      return False
    
    def quat2SO3(self,quaternion):
        w, x, y, z = quaternion
        return np.array([[w*w + x*x - y*y - z*z, 2*(x*y - z*w), 2*(w*y + x*z)],
                   [2*(w*z + x*y), w*w - x*x + y*y - z*z, 2*(y*z - w*x)],
                   [2*(x*z - w*y), 2*(w*x + y*z), w*w - x*x - y*y + z*z]])
    
    def rotate_quat_90_y(self,quaternion):
        q_y_90 = np.array([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0])  # [cos(45°), 0, sin(45°), 0]

        rotated_quat = self.quat_multiply(quaternion, q_y_90)

        return rotated_quat
    
    def box_orientation(self,p1, p2, p3, p4):
        p1, p2, p3, p4 = map(np.array, (p1, p2, p3, p4))
    
        v1 = p2 - p1  # First edge vector
        v2 = p3 - p1  # Second edge vector

        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)  # Normalize Z-axis

        if normal[2] < 0:  
            normal = -normal  
        if np.abs(normal[0]) > np.abs(normal[1]):  
            x_axis = np.cross([0, 1, 0], normal)  # Use global Y-axis
        else:
            x_axis = np.cross([1, 0, 0], normal)  # Use global X-axis

        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(normal, x_axis)
        y_axis /= np.linalg.norm(y_axis)  
        R_matrix = np.column_stack((x_axis, y_axis, normal))
        R_matrix += 1e-6 * np.eye(3)

        try:
            U, _, Vt = np.linalg.svd(R_matrix)
            R_matrix = U @ Vt  
        except np.linalg.LinAlgError:
            print("SVD failed. Using fallback method.")
            R_matrix = np.eye(3)  
        quaternion = self.SO32quat(R_matrix)

        return quaternion, normal
    
    def box_midpoint(self, p1, p2, p3, p4):
        p1, p2, p3, p4 = map(np.array, (p1, p2, p3, p4))
        midpoint = (p1 + p2 + p3 + p4) / 4.0
        return midpoint   
    
    def pos_interpolate(self,initial_pos, target_pos, current_time, total_time):
  
        t = np.clip(current_time / total_time, 0, 1)
        cos_t = (1 - np.cos(np.pi * t)) / 2
        interpolated_pos = (1 - cos_t) * np.array(initial_pos) + cos_t * np.array(target_pos)
        return interpolated_pos
    
    def ee_box_collision(self):
     
        ee_id    = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "EE_box")
        plate_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "detection_plate")

    
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            # find which bodies each geom belongs to
            b1 = self.m.geom_bodyid[c.geom1]
            b2 = self.m.geom_bodyid[c.geom2]

            # check if one end is EE and the other is the plate
            if (b1 == ee_id and b2 == plate_id) or (b1 == plate_id and b2 == ee_id):
                return True

        return False
        
    ##====================================================================
    ##         Control Utility Functions 
    ##===================================================================
    
    def SO32quat(self,R):
        """Convert a rotation matrix to a quaternion."""
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (R[2, 1] - R[1, 2]) * s
            qy = (R[0, 2] - R[2, 0]) * s
            qz = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                qw = (R[2, 1] - R[1, 2]) / s
                qx = 0.25 * s
                qy = (R[0, 1] + R[1, 0]) / s
                qz = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                qw = (R[0, 2] - R[2, 0]) / s
                qx = (R[0, 1] + R[1, 0]) / s
                qy = 0.25 * s
                qz = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                qw = (R[1, 0] - R[0, 1]) / s
                qx = (R[0, 2] + R[2, 0]) / s
                qy = (R[1, 2] + R[2, 1]) / s
                qz = 0.25 * s

        return np.array([qw, qx, qy, qz])
    
   

    def quat_multiply(self,quaternion0, quaternion1):
        w0, x0, y0, z0 = quaternion0
        w1, x1, y1, z1 = quaternion1
        v0 = np.array([x0, y0, z0])
        v1 = np.array([x1, y1, z1])
        r = w0 * w1 - v0.dot(v1)
        v = w0 * v1 + w1 * v0 + np.cross(v0, v1)
        return np.array([r, v[0], v[1], v[2]], dtype=float)
    
    def quat2so3(self,quaternion):
        w, x, y, z = quaternion
        theta = 2 * np.arccos(w)
        if abs(theta) < 0.0001:
            return np.zeros(3)
        else:
            return theta / np.sin(theta/2) * np.array([x, y, z], dtype=float)
        

    def quat_inv(self,quaternion):
        w0, x0, y0, z0 = quaternion
        return np.array([w0, -x0, -y0, -z0], dtype=float)

    