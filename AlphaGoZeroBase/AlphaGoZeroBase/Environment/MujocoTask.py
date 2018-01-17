
from Environment.MujocoModel import MujocoModel
from mujoco_py import MjSim
import numpy as np
import json

class TaskConfig:
    def __init__(self, limitTime, targetDistance):
        self.ClearScore = -0.1
        self.ClearBonusScore = 100000
        self.LimitTime = limitTime
        self.TargetDistance = targetDistance

    def GetList(self):
        return list([self.LimitTime, self.TargetDistance])

    def SetList(self, a):
        self.LimitTime = a[0]
        self.TargetDistance = a[1]

class MujocoTask:



    def __init__(self, model : MujocoModel, config:TaskConfig):

        self.Config = config
        self.Model = model
        self.Target = {}

        sim = MjSim(self.Model.MujocoModel)
        sim.step()

        joints = self.Model.JointList

        for joint in joints:
            self.Target[joint.Site] = sim.data.get_site_xpos(joint.Site) + np.array([config.TargetDistance,0,0])

            
    def Load(model:MujocoModel, fileName):
        
        with open(fileName, "rt") as f:
            a = json.load(f)
            config = TaskConfig(0,0)
            config.SetList(a)

        return MujocoTask(model, config)


    def Save(self, filePath):

        with open(filePath, "wt") as f:
            json.dump(self.Config.GetList(), f)


    def GetScore(self, sim : MjSim):

        sum = 0

        joints = self.Model.JointList

        for joint in joints:
            obs = self.GetJointObservation(sim, joint)
            sum -= obs[3];

        score = sum / len(joints)

        if score >= self.Config.ClearScore:
            return score + self.Config.ClearBonusScore - sim.data.time

        return score

    def GetJointObservation(self, sim : MjSim, joint : MujocoModel.Joint):

        target = self.Target[joint.Site]
        current = sim.data.get_site_xpos(joint.Site)

        diff = target-current;
        length = np.linalg.norm(np.array(diff))

        return [diff[0], diff[1], diff[2], length]


    def IsClear(self, sim : MjSim):

        return self.GetScore(sim) > self.Config.ClearScore


    def IsTerminate(self, sim : MjSim):
        
        return sim.data.time >= self.Config.LimitTime




