
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
from collections import deque

class Evaluater:
    def __init__(self, config:AllConfig):

        self.Config = config


    def Start(self):

        net = self.LoadNet()
        
        if net.OptimizeCount < self.Config.Worker.CheckPointLength:
            print("Optimze Count "+str(net.OptimizeCount)+" < CheckPointLength");
            return False

        self.Evaluate(net)
        return True


    def LoadNet(self):
        
        net = NetworkModel()
        net.Load(self.Config.FilePath.NextGeneration.Config, self.Config.FilePath.NextGeneration.Weight)

        return net


    def Evaluate(self, next):

        best = NetworkModel()
        best.Load(self.Config.FilePath.BestModel.Config, self.Config.FilePath.BestModel.Weight)

        bestModel = MujocoModelHumanoid()
        bestTask = MujocoTask.Load(bestModel, self.Config.FilePath.BestModel.Task)
        bestEnv = MujocoEnv(bestModel,bestTask)

        bestinit = bestEnv.GetSimState()


        nextModel = MujocoModelHumanoid()
        nextTask = MujocoTask.Load(nextModel, self.Config.FilePath.BestModel.Task)
        nextEnv = MujocoEnv(nextModel,nextTask)

        nextinit = nextEnv.GetSimState()
        

        nextWin = 0

        samplingCount = 0
        samplingOKCount = 0

        print("Buttle Start")

        for i in range(self.Config.Worker.EvaluateButtle):

            bestAgent = Agent(self.Config.EvaluateAgent, best, bestModel, bestTask)
            nextAgent = Agent(self.Config.EvaluateAgent, next, nextModel, nextTask)

            bestAction = bestAgent.SearchBestAction(bestinit)
            nextAction = nextAgent.SearchBestAction(nextinit)

            bestScore = self.GetScore(bestEnv, bestinit, bestAction)
            nextScore = self.GetScore(nextEnv, nextinit, nextAction)


            samplingTarget = int((i+1)/self.Config.Worker.EvaluateButtle * self.Config.Worker.EvaluateTimeStepSampling)
            for j in range(samplingTarget - samplingCount):

                p = random.randint(0, min(len(bestAgent.TrainData), len(nextAgent.TrainData))-1)

                bestPolicy, bestValue = next.Model.predict_on_batch(np.array([bestAgent.TrainData[p][0]]))
                nextPolicy, nextValue = next.Model.predict_on_batch(np.array([nextAgent.TrainData[p][0]]))

                samplingCount += 1
                samplingResult = (bestValue[0]-self.Config.Worker.EvaluateTimeStepEpsilon < nextValue[0]) == (bestScore <= nextScore)
                if samplingResult==True:
                    samplingOKCount+=1

            nextAgent.SaveTrainData(self.Config.GetTrainPath("next"))

            if nextScore > bestScore:
                nextWin += 1

            if nextScore == bestScore:
                nextWin += 0.5

            print("Buttle "+str(i)+" "+str(nextScore>=bestScore))


        winRate = nextWin / self.Config.Worker.EvaluateButtle
        samplingRate = samplingOKCount / samplingCount
        print("WinRate "+str(winRate))
        print("SamplingOK "+str(samplingRate))

        if winRate >= self.Config.Worker.EvaluateWinRate:

            print("!! Next Gen Win")

            next.OptimizeCount = 0
            next.Save(self.Config.FilePath.BestModel.Config, self.Config.FilePath.BestModel.Weight)

            if samplingRate >= self.Config.Worker.EvaluateTimeStepUpdateRate:

                nextLimit = nextTask.Config.LimitTime * self.Config.Worker.EvaluateTimeStepUpdateScale
                
                print("!! TimeStepUpdate "+str(nextTask.Config.LimitTime)+" -> "+str(nextLimit))
                nextTask.Config.LimitTime = nextLimit

            nextTask.Save(self.Config.FilePath.BestModel.Task)

            bestLog = self.Config.GetBestLog()
            best.Save(bestLog.Config, bestLog.Weight)
            bestTask.Save(bestLog.Task)
            
        shutil.copyfile(self.Config.FilePath.BestModel.Config, self.Config.FilePath.NextGeneration.Config)
        shutil.copyfile(self.Config.FilePath.BestModel.Weight, self.Config.FilePath.NextGeneration.Weight)
        shutil.copyfile(self.Config.FilePath.BestModel.Task, self.Config.FilePath.NextGeneration.Task)





    def GetScore(self, env, state, action):

        env.SetSimState(state)

        for act in action:
            env.Step(act)

        return env.GetScore()



