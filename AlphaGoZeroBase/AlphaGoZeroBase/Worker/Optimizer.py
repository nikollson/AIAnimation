
from Network.NetworkModel import NetworkModel
from Worker.AllConfig import AllConfig
from Environment.MujocoEnv import MujocoEnv
from Environment.MujocoModelSimple import MujocoModelSimple
from Environment.MujocoTask import MujocoTask
from Agent.Agent import Agent
import os
import random
import json
import numpy as np
from collections import deque

class Optimizer:
    def __init__(self, config:AllConfig):

        self.Config = config
        self.Data = deque()
        self.LoadedData = set()


    def Start(self):

        self.FileLoad()
        
        net = self.LoadNet()
        
        self.Optimize(net)
        self.Evaluate(net)

    
    def FileLoad(self):

        dataDir = self.Config.TrainDir
        dataList = os.listdir(dataDir)

        for i in range(len(dataList)):

            filePath = dataDir + "/" + dataList[len(dataList)-i-1]
        
            if filePath in self.LoadedData:
                break
            
            self.LoadedData.add(filePath)

            print("** File Loading ** " + filePath)

            with open(filePath, "rt") as f:
                self.Data.extend(deque(json.load(f)))
            
            for i in range(len(self.Data)-self.Config.Worker.TrainDataMax):
                self.Data.popleft()

            print("** File Loaded ** data len = "+str(len(self.Data)))
            
            if len(self.Data) >= self.Config.Worker.TrainDataMax:
                break;

    def LoadNet(self):
        
        net = NetworkModel()
        net.Load(self.Config.FilePath.NextGeneration.Config, self.Config.FilePath.NextGeneration.Weight)

        return net
    
    def Optimize(self, net):
        
        
        observeList = []
        policyList = []
        valueList = []
        index = [random.randint(0, len(self.Data)-1) for _ in range(self.Config.Worker.TrainBatchSize)]
        random.shuffle(index)

        vave = 0
        for i in index:
            vave += self.Data[i][2]/len(index)

        for i in index:
            observeList.append(self.Data[i][0])
            policyList.append(self.Data[i][1])
            valueList.append(-1 if self.Data[i][2]<vave else 1)


        observeList = np.array(observeList)
        policyList = np.array(policyList)
        valueList = np.array(valueList)

        net.Compile(self.Config.NetworkCompile())
        net.OptimizePatch(observeList, policyList, valueList)

        print("Optimize Count : "+str(net.OptimizeCount))

        net.Save(self.Config.FilePath.NextGeneration.Config, self.Config.FilePath.NextGeneration.Weight)


    def Evaluate(self, next):

        if next.OptimizeCount < self.Config.Worker.CheckPointLength:
            return
        
        best = NetworkModel()
        best.Load(self.Config.FilePath.BestModel.Config, self.Config.FilePath.BestModel.Weight)

        bestModel = MujocoModelSimple()
        bestTask = MujocoTask.Load(bestModel, self.Config.FilePath.BestModel.Task)
        bestEnv = MujocoEnv(bestModel,bestTask)

        bestinit = bestEnv.GetSimState()


        nextModel = MujocoModelSimple()
        nextTask = MujocoTask.Load(nextModel, self.Config.FilePath.BestModel.Task)
        nextEnv = MujocoEnv(nextModel,nextTask)

        nextinit = nextEnv.GetSimState()
        

        nextWin = 0

        print("Buttle Start")

        for i in range(self.Config.Worker.EvaluateButtle):

            bestAgent = Agent(self.Config.EvaluateAgent, best, bestModel, bestTask)
            nextAgent = Agent(self.Config.EvaluateAgent, next, nextModel, nextTask)

            bestAction = bestAgent.SearchBestAction(bestinit)
            nextAction = nextAgent.SearchBestAction(nextinit)

            bestScore = self.GetScore(bestEnv, bestinit, bestAction)
            nextScore = self.GetScore(nextEnv, nextinit, nextAction)

            if nextScore >= bestScore:
                nextWin += 1

            print("Buttle "+str(i)+" "+str(nextScore>=bestScore))

        winRate = nextWin / self.Config.Worker.EvaluateButtle
        print("WinRate "+str(winRate))

        if winRate >= self.Config.Worker.EvaluateWinRate:

            print("!! Next Gen Win")

            next.OptimizeCount = 0
            next.Save(self.Config.FilePath.BestModel.Config, self.Config.FilePath.BestModel.Weight)
            nextTask.Save(self.Config.FilePath.BestModel.Task)

            bestLog = self.Config.GetBestLog()
            best.Save(bestLog.Config, bestLog.Weight)
            bestTask.Save(bestLog.Task)

        os.remove(self.Config.FilePath.NextGeneration.Config)
        os.remove(self.Config.FilePath.NextGeneration.Weight)
        os.remove(self.Config.FilePath.NextGeneration.Task)





    def GetScore(self, env, state, action):

        env.SetSimState(state)

        for act in action:
            env.Step(act)

        return env.GetScore()



