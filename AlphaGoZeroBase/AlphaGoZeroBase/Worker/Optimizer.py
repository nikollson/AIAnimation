
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
import bisect
from collections import deque

class Optimizer:
    def __init__(self, config:AllConfig):

        self.Config = config
        self.Data = None
        self.DataLength = 0
        self.DataAddedIndex = deque()
        self.LoadedData = set()
        self.DataLengthList = deque()

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

        loadDataList = []

        for i in range(min(self.Config.Worker.TrainDataMax, len(dataList))):

            filePath = dataDir + "/" + dataList[len(dataList)-i-1]

            if filePath in self.LoadedData:
                break

            loadDataList.append(filePath)

        loadDataList.reverse()


        for i in range(len(loadDataList)):

            filePath = loadDataList[i]

            self.LoadedData.add(filePath)

            print("** File Loading ** " + filePath)

            while os.access(filePath, os.R_OK)==False:
                time.sleep(0.001)

            with open(filePath, "rt") as f:

                fileData = json.load(f)

                for d in fileData:
                    self.DataLength += 1
                    
                    addIndex = np.argmax(d[1])
                    
                    self.Data[addIndex].append(d)
                    self.DataAddedIndex.append(addIndex)

                self.DataLengthList.append(len(fileData))

                if isFirst==False:
                    self.TrainCount += 1

            print("** File Loaded ** data len = "+str(self.DataLength))
            
        eraseCount = max(0, len(self.DataLengthList) - self.Config.Worker.TrainDataMax)

        for i in range(eraseCount):

            eraseLen = self.DataLengthList[0]
            self.DataLengthList.popleft()

            for j in range(eraseLen):
                self.DataLength -= 1
                    
                addIndex = self.DataAddedIndex[0]

                self.Data[addIndex].popleft()
                self.DataAddedIndex.popleft()


    def LoadNet(self):
        
        net = NetworkModel(False)
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

        enablePolicy = []
        for i in range(len(self.Data)):
            if len(self.Data[i])!=0:
                enablePolicy.append(i)

        for i in range(batchN):

            p = random.randint(0, inputN-1)

            if len(self.Data[p])==0:
                policy = np.zeros(inputN)
                policy[p] = 1

                scoreP = random.choice(enablePolicy)
                score = self.Data[scoreP][random.randint(0, len(self.Data[scoreP])-1)][2]

                observe = []
                for j in range(observeN1):
                    observeP = random.choice(enablePolicy)
                    observeP2 = random.randint(0, len(self.Data[observeP])-1)
                    observe.append(self.Data[observeP][observeP2][0][j])

            else:
                q = random.randint(0, len(self.Data[p])-1)
                policy = self.Data[p][q][1]
                observe = self.Data[p][q][0]
                score = self.Data[p][q][2]

            self.ObserveList[i] = np.array(observe)
            self.PolicyList[i] = np.array(policy)
            self.ValueList[i] = score


        sortedValue = list(self.ValueList)
        sortedValue.append(-1000000000)
        sortedValue.sort()

        for i in range(len(self.ValueList)):
            relu = self.Config.Worker.OptimizeReluEdge
            insertPer = (bisect.bisect_left(sortedValue, self.ValueList[i])-1) / (len(self.ValueList)-1)
            insertPer = min(1, max(0, (insertPer-relu)/(1-2*relu)))*2-1

            self.ValueList[i] = insertPer

        compileParam = self.Config.NetworkCompile(net.OptimizeCount)
        print("Compile "+str(compileParam.LearningRate))
        net.Compile(compileParam)

        for i in range(self.Config.Worker.TrainLoop):
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



