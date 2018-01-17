
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
from collections import deque

class Optimizer:
    def __init__(self, config:AllConfig):

        self.Config = config
        self.Data = None
        self.DataAddedIndex = deque()
        self.DataLength = 0
        self.LoadedData = set()

        self.ObserveList = -1
        self.PolicyList = -1
        self.ValueList = -1
        self.TrainCount = 0


    def Start(self):
        
        net = self.LoadNet()
        self.FileLoad(net)

        if net.OptimizeCount >= self.Config.Worker.CheckPointLength:
            print("Optimze Count "+str(net.OptimizeCount)+" >= CheckPointLength");
            return

        self.Optimize(net)

    
    def FileLoad(self, net:NetworkModel):

        dataDir = self.Config.TrainDir
        dataList = os.listdir(dataDir)
        inputN = net.Model.input_shape[1] * 2 + 1

        isFirst = False

        if self.Data == None:

            self.Data = []
            
            for i in range(inputN):
                self.Data.append(deque())

            isFirst = True


        for i in range(len(dataList)):

            filePath = dataDir + "/" + dataList[len(dataList)-i-1]

            if filePath in self.LoadedData:
                break
            
            self.LoadedData.add(filePath)

            print("** File Loading ** " + filePath)

            while os.access(filePath, os.R_OK)==False:
                time.sleep(0.001)

            with open(filePath, "rt") as f:

                fileData = json.load(f)

                for d in fileData:
                    self.DataLength += 1
                    
                    addIndex = np.argmax(d[2])
                    
                    self.Data[addIndex].append(d)
                    self.DataAddedIndex.append(addIndex)

                for i in range(self.DataLength-self.Config.Worker.TrainDataMax):
                    self.DataLength -= 1
                    
                    addIndex = self.DataAddedIndex[0]

                    self.Data[addIndex].popleft()
                    self.DataAddedIndex.popleft()

                self.TrainCount += 1

            print("** File Loaded ** data len = "+str(self.DataLength))
            
            if isFirst and self.DataLength >= self.Config.Worker.TrainDataMax:
                break;

    def LoadNet(self):
        
        net = NetworkModel()
        net.Load(self.Config.FilePath.NextGeneration.Config, self.Config.FilePath.NextGeneration.Weight)

        return net
    
    def Optimize(self, net):
        
        index = [random.randint(0, len(self.Data)-1) for _ in range(self.Config.Worker.TrainBatchSize)]
        random.shuffle(index)

        batchN = self.Config.Worker.TrainBatchSize
        inputN = net.Model.input_shape[1] * 2 + 1
        observeN1 = net.Model.input_shape[1]
        observeN2 = net.Model.input_shape[2]

        if isinstance(self.ObserveList, np.ndarray)==False:
            self.ObserveList = np.ndarray((batchN, observeN1, observeN2))
            self.PolicyList = np.ndarray((batchN, inputN))
            self.ValueList = np.ndarray(batchN)

        for i in range(batchN):

            p = random.randint(0, inputN-1)

            if len(self.Data[p])==0:
                policy = np.zeros(inputN)
                policy[p] = 1

                for _ in range(100000):
                    t = random.randint(0, inputN-1)

                    if len(self.Data[t])!=0:
                        observe = self.Data[t][random.randint(0, len(self.Data[t])-1)][0]
                        score = self.Data[t][random.randint(0, len(self.Data[t])-1)][2]
                        break;
            else:
                q = random.randint(0, len(self.Data[p])-1)
                policy = self.Data[p][q][1]
                observe = self.Data[p][q][0]
                score = self.Data[p][q][2]

            self.ObserveList[i] = np.array(observe)
            self.PolicyList[i] = np.array(policy)
            self.ValueList[i] = score

        copy = list(self.ValueList)
        copy.sort()

        centerScore = copy[int(len(copy)/2)]

        for i in range(len(self.ValueList)):
            self.ValueList[i] = -1 if self.ValueList[i]<centerScore else 1

        net.Compile(self.Config.NetworkCompile(net.OptimizeCount))
        net.OptimizePatch(self.ObserveList, self.PolicyList, self.ValueList)

        net.OptimizeCount += self.TrainCount
        self.TrainCount = 0

        net.Save(self.Config.FilePath.NextGeneration.Config, self.Config.FilePath.NextGeneration.Weight)

        print("Optimize Count : "+str(net.OptimizeCount))


    def GetScore(self, env, state, action):

        env.SetSimState(state)

        for act in action:
            env.Step(act)

        return env.GetScore()



