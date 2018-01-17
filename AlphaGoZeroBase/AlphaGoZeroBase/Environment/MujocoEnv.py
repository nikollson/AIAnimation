

from Environment.MujocoModel import MujocoModel
from Environment.MujocoTask import MujocoTask, TaskConfig
from mujoco_py import MjSim, MjViewer
import numpy as np


class MujocoEnv:


    def __init__(self, model : MujocoModel, task : MujocoTask):
        
        self.Model = model;
        self.Sim = MjSim(self.Model.MujocoModel)

        self.Task = task
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


    def Render(self):

        if self.Viewer is None:
            self.Viewer = MjViewer(self.Sim)

        self.Viewer.render()


    def GetObservation(self):
        
        joints = self.Model.JointList

        N = len(joints)

        ret = None

        for i in range(N):

            observe = []

            observe.extend(self.GetSensorValue(joints[i].JointPosition))
            observe.extend(self.GetSensorValue(joints[i].JointVelocity))
            observe.extend(self.GetSensorValue(joints[i].Accel))
            observe.extend(self.GetSensorValue(joints[i].Force))
            observe.extend(self.GetSensorValue(joints[i].Gyro))
            observe.extend(self.GetSensorValue(joints[i].Torque))
            observe.extend(self.GetSensorValue(joints[i].Velocity))

            observe.extend(self.Task.GetJointObservation(self.Sim, joints[i]))

            if i==0:
                ret = np.zeros((N, len(observe)))

            ret[i] = observe

        return ret


    def GetSensorValue(self, sensorName : str):

        if sensorName == "":
            return 0
        id = self.Sim.model.sensor_name2id(sensorName)
        adr = self.Sim.model.sensor_adr[id]
        dim = self.Sim.model.sensor_dim[id]

        ret = []

        for i in range(dim):
            ret.append(self.Sim.data.sensordata[adr+i])

        return ret
    

    def GetObservationShape(self):

        return self.GetObservation().shape
    

    def GetActionNum(self):

        return self.Model.NAction


    def IsTerminate(self):

        return self.Task.IsTerminate(self.Sim) or self.Task.IsClear(self.Sim)

    def GetScore(self):

        return self.Task.GetScore(self.Sim)

    def GetTime(self):

        return self.Sim.data.time