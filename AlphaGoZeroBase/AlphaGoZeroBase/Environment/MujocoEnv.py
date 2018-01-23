

from Environment.MujocoModel import MujocoModel
from Environment.MujocoTask import MujocoTask, TaskConfig
from mujoco_py import MjSim, MjViewer
import numpy as np
import math


class MujocoEnv:


    def __init__(self, model : MujocoModel):
        
        self.Model = model;
        self.Sim = MjSim(self.Model.MujocoModel)

        self.Viewer = None


    def GetSimState(self):

        return self.Sim.get_state()
    

    def SetSimState(self, state):

        self.Sim.set_state(state)


    def Step(self, actionNum = None):
        
        if actionNum == None:
            actionNum = self.Model.NAction - 1;

        torque = self.Model.GetActionTorque(actionNum)

        for i in range(len(torque)):
            self.Sim.data.ctrl[i] = torque[i]
            
        self.Sim.step()
        self.Sim.step()
        self.Sim.step()


    def Render(self):

        if self.Viewer is None:
            self.Viewer = MjViewer(self.Sim)

        self.Viewer.render()


    def GetObservation(self, task:MujocoTask):
        
        joints = self.Model.JointList

        N = len(joints)

        ret = None

        for i in range(N):
            observe = []

            observe.extend(self.GetSensorValue(1, joints[i].JointPosition))
            observe.extend(self.GetSensorValue(1, joints[i].JointVelocity))
            observe.extend(self.GetSensorValue(3, joints[i].Accel))
            observe.extend(self.GetSensorValue(3, joints[i].Gyro))
            observe.extend(self.GetSensorValue(3, joints[i].Torque))
            observe.extend(self.GetSensorValue(3, joints[i].Velocity))

            observe.extend(self.MatToAngle(self.Sim.data.get_site_xmat(joints[i].Site)))

            observe.extend(task.GetJointObservation(self.Sim, joints[i]))
            
            if i==0:
                ret = np.zeros((N, len(observe)))

            ret[i] = observe

        return ret


    def GetSensorValue(self, dim, sensorName : str):

        if (sensorName in self.Sim.model.sensor_names) == False:
            ret = []
            for _ in range(dim):
                ret.append(0)
            return ret

        id = self.Sim.model.sensor_name2id(sensorName)
        adr = self.Sim.model.sensor_adr[id]

        ret = []

        for i in range(dim):
            ret.append(self.Sim.data.sensordata[adr+i])

        return ret
    
    def GetObservationShape(self, task):

        return self.GetObservation(task).shape
    

    def GetActionNum(self):

        return self.Model.NAction

    
    def MatToAngle(self, m):
        return [math.asin(m[2,1]), math.atan2(-m[0,1],m[1,1]), math.atan2(-m[2,0],m[2,2])]

    def IsTerminate(self, task:MujocoTask, score, timeLimit):

        return task.IsClear(score) or timeLimit <= self.Sim.data.time

    def GetScore(self, task:MujocoTask):

        return task.GetScore(self.Sim)

    def GetTime(self):

        return self.Sim.data.time
