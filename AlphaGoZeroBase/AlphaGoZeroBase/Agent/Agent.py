

from Environment.MujocoEnv import MujocoEnv
from Environment.MujocoModel import MujocoModel
from Network.NetworkModel import NetworkModel

import numpy as np
import json


class AgentConfig:

    def __init__(self, searchAmount, beamWidth, tau):

        self.SearchAmount = searchAmount
        self.BeamWidth = beamWidth
        self.SearchDepthMax = 1000
        self.CPuct = 5
        self.DiriclhetAlpha = 0.03
        self.DiriclhetEpsilon = 0.25
        self.PolicyTau = tau
        self.PolicyTauMaxTime = 0.5

    def GetTau(self, time, maxTime):

        if time <= self.PolicyTauMaxTime:
            return self.PolicyTau

        par = 1 - (time-self.PolicyTauMaxTime) / (maxTime-self.PolicyTauMaxTime)

        return  par * self.PolicyTau


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


    def Expand(self, network:NetworkModel, env:MujocoEnv):

        env.SetSimState(self.Parent.State)
        env.Step(self.ActionNum)

        self.State = env.GetSimState()
        self.Observation = env.GetObservation()
        self.IsTerminate = env.IsTerminate()
        self.Score = env.GetScore()

        policy_arr, value_arr = network.Model.predict(np.array([self.Observation]))

        policy = policy_arr[0] / np.sum(policy_arr[0])
        value = value_arr[0][0]

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
            nn = self.Children[i].N
            if self.PickedList[i]==True:
                nn=0
            nList.append(nn)
            aList.append(i)
        
        if tau <= 0.00001:
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
            self.PickedPolicy = nList
        
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
        self.Env = MujocoEnv(model, task)
        self.StepTarget = []
        self.TrainData = list([])


    def SearchBestAction(self, initialState):

        bestAction = []
        
        initialNode = RootNode([], initialState)

        firstNode = Node(initialNode, 1, self.Env.Model.NoneAction)
        firstNode.Expand(self.Network, self.Env)

        self.StepTarget.append([firstNode])
        

        for i in range(self.Config.SearchDepthMax):
            
            if i%10==0:
                print("Simuration Step "+str(i))

            isEnd = False

            for node in self.StepTarget[i]:
                isEnd |= node.IsTerminate

            if isEnd:
                break
            
            searchRoot = RootNode(self.StepTarget[i], None)
            self.StepTarget.append([])

            while True:
                value = self.SearchMoves(searchRoot, 1)
                
                if value == None:
                    break

                for child in searchRoot.Children:
                    if child.N >= self.Config.SearchAmount:
                        tau = self.Config.GetTau(self.Env.GetTime(), self.Env.Task.GetLimit())
                        self.StepTarget[i+1].append(child.PickTopChild(tau))

                if len(self.StepTarget[i+1]) >= self.Config.BeamWidth:
                    break


        resultNodes = self.StepTarget[len(self.StepTarget)-1]
        resultNodes.sort(key=lambda x:x.Score, reverse=True)

        resultCount = len(resultNodes)

        for i in range(resultCount):

            win = True if i < resultCount/2 else False

            trainData = self.MakeTrainData(resultNodes[i], resultNodes[i].Score)

            self.TrainData.extend(trainData)

            

            if i==0:
                bestAction = self.GetActionList(resultNodes[i])
                print("result "+str(i)+" "+str(resultNodes[i].Score))
                print(bestAction)

        return bestAction


    def SearchMoves(self, node, noiseEnable):

        assert isinstance(node, Node)
        
        if node.IsExpanded == False:
            value = node.Expand(self.Network, self.Env)
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



