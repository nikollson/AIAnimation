
from Network.NetworkModel import NetworkModel
from Worker.AllConfig import AllConfig
from Environment.MujocoEnv import MujocoEnv
from Environment.MujocoModelHumanoid import MujocoModelHumanoid
from Environment.MujocoTask import MujocoTask
from Agent.Agent import Agent
import os
import random
import json
import numpy as np
import shutil
import time
from collections import deque

class Evaluater:
    def __init__(self, config:AllConfig):

        self.Config = config


    def Start(self):

        next = self.LoadNet()
        
        if next.OptimizeCount < self.Config.Worker.CheckPointLength:
            print("Optimze Count "+str(next.OptimizeCount)+" < CheckPointLength");
            return False
        
        best = NetworkModel()
        best.Load(self.Config.FilePath.BestModel.Config, self.Config.FilePath.BestModel.Weight)

        self.EvaluateToBest(best, next)
        return True


    def LoadNet(self):
        
        net = NetworkModel()
        net.Load(self.Config.FilePath.NextGeneration.Config, self.Config.FilePath.NextGeneration.Weight)

        return net


    def EvaluateToBest(self, best, next):

        dataList = os.listdir(self.Config.Task.EvalDir)
        
        nextWin = 0
        
        print("Buttle Start")

        for i in range(len(dataList)):
            dataName = dataList[i]

            win = self.IsNextWin(best, next, self.Config.Task.EvalDir+"/"+dataName)
            nextWin += 1 if win else 0

            print("Buttle "+str(i)+" "+str(win))
            print()

        
        winRate = nextWin / len(dataList)
        print("WinRate "+str(winRate))

        if winRate >= self.Config.Worker.EvaluateWinRate:

            print("!! Next Gen Win")

            next.OptimizeCount = 0
            next.TimeLimit *= self.Config.Worker.EvaluateTimeStepExpand

            while True:
                try:
                    next.Save(self.Config.FilePath.BestModel.Config, self.Config.FilePath.BestModel.Weight)
                    break
                except:
                    time.sleep(0.1)

            while True:
                try:
                    trainDataList = os.listdir(self.Config.TrainDir)
                    for i in trainDataList:
                        os.remove(self.Config.TrainDir+"/"+i)
                    break
                except:
                    time.sleep(0.1)

            bestLog = self.Config.GetBestLog()
            best.Save(bestLog.Config, bestLog.Weight)

        shutil.copyfile(self.Config.FilePath.BestModel.Config, self.Config.FilePath.NextGeneration.Config)
        shutil.copyfile(self.Config.FilePath.BestModel.Weight, self.Config.FilePath.NextGeneration.Weight)



    def IsNextWin(self, best, next, filePath):

        bestModel = MujocoModelHumanoid()
        bestTask = MujocoTask(bestModel, filePath)
        bestEnv = MujocoEnv(bestModel)


        nextModel = MujocoModelHumanoid()
        nextTask = MujocoTask(nextModel, filePath)
        nextEnv = MujocoEnv(nextModel)

        bestAgent = Agent(self.Config.EvaluateAgent, best, bestModel, bestTask)
        nextAgent = Agent(self.Config.EvaluateAgent, next, nextModel, nextTask)

        bestAction = bestAgent.SearchBestAction()
        nextAction = nextAgent.SearchBestAction()

        bestScore = self.GetScore(bestEnv, bestTask, bestAction)
        nextScore = self.GetScore(nextEnv, nextTask, nextAction)

        nextAgent.SaveTrainData(self.Config.GetTrainPath("next"))

        return nextScore > bestScore




    def GetScore(self, env, task, action):

        env.SetSimState(task.StartState)

        for act in action:
            env.Step(act)

        return env.GetScore(task)



