
from Environment.MujocoModel import MujocoModel
from mujoco_py import MjSim
import numpy as np

class Config:
    ClearScore = -1.0

class MujocoTask:
    def __init__(self, model : MujocoModel, limitTime, targetDistance):

        self.Model = model
        self.LimitTime = limitTime

        self.Target = {}

        sim = MjSim(self.Model.MujocoModel)
        sim.step()

        joints = self.Model.JointList

        for joint in joints:
            self.Target[joint.Site] = sim.data.get_site_xpos(joint.Site) + np.array([0,0,-targetDistance])

    def GetScore(self, sim : MjSim):

        sum = 0

        joints = self.Model.JointList

        for joint in joints:
            obs = self.GetJointObservation(sim, joint)
            sum -= obs[3];

        return sum

    def GetJointObservation(self, sim : MjSim, joint : MujocoModel.Joint):

        target = self.Target[joint.Site]
        current = sim.data.get_site_xpos(joint.Site)

        diff = target-current;
        length = np.linalg.norm(np.array(diff))

        return [diff[0], diff[1], diff[2], length]


    def IsClear(self, sim : MjSim):

        return self.GetScore(sim) > Config.ClearScore


    def IsTerminate(self, sim : MjSim):
        
        return sim.data.time >= self.LimitTime




