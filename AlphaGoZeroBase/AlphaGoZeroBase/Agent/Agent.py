

from Environment.MujocoEnv import MujocoEnv
from Environment.MujocoModel import MujocoModel
from Network.NetworkModel import NetworkModel

import numpy as np
import json
import bisect

class AgentConfig:

    def __init__(self, valueCalc, searchAmount, beamWidth, tau, endTau, epsilon):

        self.ValueCalc = valueCalc
        self.SearchAmount = searchAmount
        self.BeamWidth = beamWidth
        self.SearchDepthMax = 10000
        self.CPuct = 2.5
        self.DiriclhetAlpha = 0.03
        self.DiriclhetEpsilon = epsilon
        self.PolicyTau = tau
        self.PolicyEndTau = endTau
        self.PolicyTauMaxTime = 0.3


    def GetTau(self, time, maxTime):

        if time/maxTime <= self.PolicyTauMaxTime:
            return self.PolicyTau

        par = 1 - (time/maxTime - self.PolicyTauMaxTime)/(1-self.PolicyTauMaxTime)

        return  par * (self.PolicyTau-self.PolicyEndTau) + self.PolicyEndTau

class ValueCalcConfig:
    def __init__(self, valueSaveMax):
        self.ValueCalculatorPath = "ValueCalc.txt"
        self.ValueSaveMax = valueSaveMax

class ValueCaluculator:
    def __init__(self, config:ValueCalcConfig):
        self.FilePath = config.ValueCalculatorPath
        self.MaxLength = config.ValueSaveMax
        self.Data = [-10000000]
        
        try:
            with open(self.FilePath, "rt") as f:
                self.Data = json.load(f)
        except:
            pass

    def CalcValue(self, score):
        insertPos = bisect.bisect_right(self.Data, score)
        return insertPos / len(self.Data)
    
    def AppendScore(self, score):
        insertPos = bisect.bisect_right(self.Data, score)
        self.Data.insert(insertPos, score)

        if len(self.Data)>=self.MaxLength:
            self.Data.pop(0)
        
        with open(self.FilePath, "wt") as f:
            json.dump(self.Data, f)
        

class Node:

    def __init__(self, parent, transitionP, actionNum):
            
        self.IsExpanded = False
        self.IsTerminate = False
        self.Score = 0
        self.Children = []
        self.Parent = parent
        self.State = None
        self.Observation = None
        self.PickedPolicy = None
        self.PickedList = None
        self.N = 0
        self.W = 0
        self.Q = 0
        self.P = transitionP
        self.ActionNum = actionNum


    def Expand(self, network:NetworkModel, env:MujocoEnv, task, valueCalc):

        env.SetSimState(self.Parent.State)
        env.Step(self.ActionNum)

        self.State = env.GetSimState()
        self.Observation = env.GetObservation(task, network.TimeLimit)
        self.Score = env.GetScore(task)
        self.IsTerminate = env.IsTerminate(task, self.Score, network.TimeLimit)

        policy_arr, value_arr = network.Model.predict(np.array([self.Observation]))

        policy = policy_arr[0]
        
        value = np.sum(value_arr[0])/len(value_arr[0])

        if self.IsTerminate == True:
            value = valueCalc.CalcValue(self.Score)

        for i in range(len(policy)):
            self.Children.append(Node(self, policy[i], i))

        self.IsExpanded = True
        self.N = 1
        self.W = value
        self.Q = value

        return value


    def PickTopChild(self, tau):

        if self.PickedList == None:
            self.PickedList = [False for _ in range(len(self.Children))]

        nList = []
        aList = []
        for i in range(len(self.Children)):
            nn = self.Children[i].N * 100000 + self.Children[i].Q
            if self.PickedList[i]==True:
                nn=0
            nList.append(nn)
            aList.append(i)
        
        if tau <= 0.05:
            maxi = np.argmax(nList)
            for i in range(len(nList)):
                if i==maxi:
                    nList[i]=1
                else:
                    nList[i]=0
        else:
            sum = 0
            for i in range(len(nList)):
                nList[i] = np.power(nList[i], 1/tau)
                sum += nList[i]

            for i in range(len(nList)):
                nList[i] /= sum
            
        action = np.random.choice(aList, p=nList)

        self.PickedList[action] = True

        if self.PickedPolicy==None:
            addList = []
            for i in range(len(self.Children)):
                addList.append(1 if i==action else 0)
            '''
            sum = 0
            for i in range(len(self.Children)):
                cn = self.Children[i].N
                sum += cn
                addList.append(cn)
            for i in range(len(addList)):
                addList[i]/=sum
            '''
            self.PickedPolicy = addList
        
        self.N -= self.Children[action].N;
        self.W -= self.Children[action].W;
        self.Q = self.W / self.N;

        return self.Children[action]


    def GetBestAction_PUCT(self, cPuct, dirichletAlpha, dirichletEpsilon, doAddNoise):

        sumN = np.sqrt(self.N)

        vlist = []
        ilist = []

        noise = np.zeros(len(self.Children))

        if doAddNoise:
            noise = np.random.dirichlet([dirichletAlpha for _ in range(len(self.Children))]) * dirichletEpsilon

        for i in range(len(self.Children)):

            if self.PickedList!=None and self.PickedList[i]==True:
                continue

            child = self.Children[i]

            p = child.P + noise[i]

            u = cPuct * p * sumN / (1 + child.N)

            v = child.Q + u
           
            vlist.append(child.Q + u)
            ilist.append(i)
        
        if len(vlist) == 0:
            return None

        return ilist[np.argmax(vlist)]


class RootNode(Node):
    def __init__(self, children, state):
        super().__init__(None, 1, None)

        self.IsExpanded = True
        self.Children = children
        self.State = state

        for child in self.Children:
            self.N += child.N
            self.W += child.W
            child.P = 1 / len(children)

        if self.N != 0:
            self.Q = self.W/self.N

class Agent:

    def __init__(self, config, network, model, task):
        
        assert isinstance(network, NetworkModel)
        assert isinstance(model, MujocoModel)

        self.Config = config
        self.Network = network
        self.Env = MujocoEnv(model)
        self.Task = task
        self.StepTarget = []
        self.TrainData = list([])
        self.ValueCalclater = ValueCaluculator(config.ValueCalc)


    def SearchBestAction(self):

        bestAction = []
        
        initialNode = RootNode([], self.Task.StartState)

        firstNode = Node(initialNode, 1, self.Env.Model.NoneAction)
        firstNode.Expand(self.Network, self.Env, self.Task, self.ValueCalclater)

        self.StepTarget.append([firstNode])
        
        print("Simuration Step ", end =" ")

        for i in range(self.Config.SearchDepthMax):
            
            if i%20==0:
                print(str(i), end=" ", flush=True)

            isEnd = False

            for node in self.StepTarget[i]:
                isEnd |= node.IsTerminate

            if isEnd:
                break
            
            searchRoot = RootNode(self.StepTarget[i], None)
            self.StepTarget.append([])

            while True:
                value = self.SearchMoves(searchRoot, 6)
                
                if value == None:
                    break

                for child in searchRoot.Children:
                    if child.N >= self.Config.SearchAmount:
                        tau = self.Config.GetTau(self.Env.GetTime(), self.Network.TimeLimit)
                        self.StepTarget[i+1].append(child.PickTopChild(tau))

                if len(self.StepTarget[i+1]) >= self.Config.BeamWidth:
                    break
        print("End")

        resultNodes = self.StepTarget[len(self.StepTarget)-1]
        resultNodes.sort(key=lambda x:x.Score, reverse=True)

        resultCount = len(resultNodes)

        for i in range(resultCount):

            win = True if i < resultCount/2 else False

            trainData = self.MakeTrainData(resultNodes[i], resultNodes[i].Score)

            self.TrainData.extend(trainData)

            if i==0:
                bestAction = self.GetActionList(resultNodes[i])
                value = self.ValueCalclater.CalcValue(resultNodes[i].Score)
                self.ValueCalclater.AppendScore(resultNodes[i].Score)
                print("result "+str(i)+" Score="+str(resultNodes[i].Score)+"  Value="+str(value))

        return bestAction


    def SearchMoves(self, node, noiseEnable):

        assert isinstance(node, Node)
        
        if node.IsExpanded == False:
            value = node.Expand(self.Network, self.Env, self.Task, self.ValueCalclater)
            return value

        action = node.GetBestAction_PUCT(self.Config.CPuct, self.Config.DiriclhetAlpha, 
                                         self.Config.DiriclhetEpsilon, noiseEnable>0)

        if action == None:
            return None

        value = self.SearchMoves(node.Children[action], noiseEnable-1)

        if value == None:
            return None

        node.N += 1
        node.W += value
        node.Q = node.W / node.N

        return value


    def MakeTrainData(self, node, value):

        assert isinstance(node, Node)

        trainData = list([])

        while True:
            
            node = node.Parent

            if node.Parent == None:
                break

            if node.PickedPolicy==None :
                continue

            policy = node.PickedPolicy

            trainData.append(list([node.Observation.tolist(), policy, value]))

        trainData.reverse()

        return trainData


    def GetActionList(self, node):

        assert isinstance(node, Node)

        actionList = []

        while True:
            
            if node.Parent == None:
                break
            
            actionList.append(node.ActionNum)
            node = node.Parent


        actionList.reverse()

        return actionList

    def SaveTrainData(self, path):
        
        with open(path, "wt") as f:
            json.dump(self.TrainData, f)



