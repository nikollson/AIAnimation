
from Environment.MujocoModel import MujocoModel
from mujoco_py import MjSim
import numpy as np
import json
import os
import random
import math
import bisect

class TaskConfig:
    def __init__(self, fileName):
        self.ClearScore = -0.1
        self.ClearBonusScore = 10000

        self.angleScale = 1/math.pi
        
        with open(fileName, "rt") as f:
            stats = json.load(f)
            
        self.StartConfig = stats[0]
        self.EndConfig = stats[1]



class MujocoTask:

    def __init__(self, model : MujocoModel, fileName):

        self.FileName = fileName
        self.Model = model
        self.Config = TaskConfig(fileName)


        sim = MjSim(self.Model.MujocoModel)

        self.StartState = self.MakeState(sim, self.Config.StartConfig)
        self.EndState = self.MakeState(sim, self.Config.EndConfig)


        self.TargetPos = {}
        self.TargetAngle = {}
        
        joints = self.Model.JointList
        
        sim.set_state(self.EndState)
        sim.step()

        for joint in joints:
            self.TargetPos[joint.Site] = np.array(sim.data.get_site_xpos(joint.Site))
            self.TargetAngle[joint.Site] = np.array(self.MatToAngle(sim.data.get_site_xmat(joint.Site)))


    def MakeState(self, sim, stateConfig):

        state = sim.get_state()

        for k,v in stateConfig.items():
            state.qpos[sim.model.get_joint_qpos_addr(k)] = v
            
        return state


    def LoadRandom(model, dir):
        
        dataList = os.listdir(dir)
        return MujocoTask(model, dir+"/"+random.choice(dataList))
   
    def MatToAngle(self, m):
        return [math.asin(m[2,1]), math.atan2(-m[0,1],m[1,1]), math.atan2(-m[2,0],m[2,2])]

    def DiffAngle(self, a, b):
        diff = a-b
        for i in range(len(diff)):
            if diff[i]>=math.pi:
                diff[i] -= math.pi*2
            if diff[i]<=-math.pi:
                diff[i] += math.pi*2
        return diff

    def GetScore(self, sim : MjSim):

        sum = 0

        joints = self.Model.JointList

        for joint in joints:
            obs = self.GetJointObservation(sim, joint)
            sum -= obs[3]
            sum -= obs[7]

        score = sum / len(joints)

        if self.IsClear(score):
            return self.Config.ClearBonusScore - sim.data.time

        return score

    def GetJointObservation(self, sim : MjSim, joint : MujocoModel.Joint):

        targetPos = self.TargetPos[joint.Site]
        currentPos = sim.data.get_site_xpos(joint.Site)

        diffPos = targetPos-currentPos
        length = np.linalg.norm(np.array(diffPos))

        
        targetAngle = self.TargetAngle[joint.Site]
        currentAngle = self.MatToAngle(sim.data.get_site_xmat(joint.Site))

        diffAngle = self.DiffAngle(targetAngle,currentAngle)
        angleSum = math.fabs(diffAngle[0]) + math.fabs(diffAngle[1]) + math.fabs(diffAngle[2]) 
        angleSum *= self.Config.angleScale

        return [diffPos[0], diffPos[1], diffPos[2], length,
                diffAngle[0], diffAngle[1], diffAngle[2], angleSum]


    def IsClear(self, score):
        return score >= self.Config.ClearScore
