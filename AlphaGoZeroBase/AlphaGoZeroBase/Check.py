
from Network.NetworkModel import NetworkModel
from Worker.AllConfig import AllConfig
from Environment.MujocoEnv import MujocoEnv
from Environment.MujocoModelHumanoid import MujocoModelHumanoid
from Environment.MujocoTask import MujocoTask
from Agent.Agent import Agent
from Worker.Logger import Logger
import os
import random
import json
import numpy as np
import shutil
import time
from collections import deque

class Checker:
    def __init__(self, config:AllConfig, logName):

        self.Config = config
        self.Logger = Logger(logName)


    def Start(self, pathcnf, pathwgt, timeLimit=-1):

        net = self.LoadNet(pathcnf, pathwgt)
        
        if timeLimit!=-1:
            net.TimeLimit = timeLimit
        
        self.Check(net, pathcnf)
        return True


    def LoadNet(self, pathcnf, pathwgt):
        
        net = NetworkModel()
        net.Load(pathcnf, pathwgt)

        return net


    def Check(self, net, name):

        dataList = os.listdir(self.Config.Task.EvalDir)
        
        clearCount = 0
        sum = 0
        
        print("Check Start")

        for i in range(len(dataList)):

            dataName = dataList[i]

            score = self.CalcScore(net, self.Config.Task.EvalDir+"/"+dataName)
            
            sum += min(0, score)

            clear = 1 if score >= 0 else 0
            clearCount += clear

            print("Check "+name+" "+str(i)+" "+str(clear)+"  "+str(clearCount)+"/"+str(i+1)+"  sum="+str(sum))
            print()
            self.Logger.AddLog("Check "+name+" "+dataName+" "+("win" if clear!=0 else "lose")+" "+str(score))
        
        clearRate = clearCount / len(dataList)
        print("ClearRate "+str(clearRate))

        self.Logger.AddLog("CheckEnd "+name+" "+str(clearRate)+" "+str(sum))

    def CalcScore(self, net, filePath):

        bestModel = MujocoModelHumanoid()
        bestTask = MujocoTask(bestModel, filePath)
        bestEnv = MujocoEnv(bestModel)


        bestAgent = Agent(self.Config.CheckerAgent, net, bestModel, bestTask)

        bestAction = bestAgent.SearchBestAction()

        bestScore = self.GetScore(bestEnv, bestTask, bestAction)

        return bestScore




    def GetScore(self, env, task, action):

        env.SetSimState(task.StartState)

        for act in action:
            env.Step(act)

        return env.GetScore(task)
    
timeLimit = 1.2

config = AllConfig()
checker = Checker(config, "Checker12-10")

dataList = os.listdir(config.BestLogDir)

for i in range(len(dataList)):
    if i%2==1:
        continue
    checker.Start(config.BestLogDir+"/"+dataList[i], config.BestLogDir+"/"+dataList[i+1], timeLimit)

