

from Environment.MujocoEnv import MujocoEnv
from Environment.MujocoModel import MujocoModel
from Network.NetworkModel import NetworkModel

import numpy as np



class AgentConfig:

    def __init__(self, searchAmount, beamWidth):

        self.SearchAmount = searchAmount
        self.BeamWidth = beamWidth
        self.SearchDepthMax = 1000
        self.CPuct = 5
        self.DiriclhetAlpha = 0.5
        self.PlicyTau = 1

class Node:

    def __init__(self, parent, transitionP, actionNum):
            
        self.IsExpanded = False
        self.IsTerminate = False
        self.Score = 0
        self.Children = []
        self.Parent = parent
        self.State = None
        self.Observation = None
        self.PickedTransition = None
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


    def PickTopChild(self):

        if self.PickedTransition == None:
            self.PickedTransition = [child.N for child in self.Children]
            self.PickedList = [False for _ in range(len(self.Children))]

        nList = []
        aList = []
        for i in range(len(self.Children)):
            if self.PickedList[i]==False:
                nList.append(self.Children[i].N)
                aList.append(i)

        action = aList[np.argmax(nList)]

        self.PickedList[action] = True

        self.N -= self.Children[action].N;
        self.W -= self.Children[action].W;
        self.Q = self.W / self.N;

        return self.Children[action]


    def GetBestAction_PUCT(self, cPuct, dirichletAlpha, doAddNoise):

        sumN = np.sqrt(self.N)

        vlist = []
        ilist = []

        noise = np.zeros(len(self.Children))

        if doAddNoise:
            noise = np.random.dirichlet([dirichletAlpha for _ in range(len(self.Children))])

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

    def __init__(self, config, network, model):
        
        assert isinstance(network, NetworkModel)
        assert isinstance(model, MujocoModel)

        self.Config = config
        self.Network = network
        self.Env = MujocoEnv(model)
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
                        self.StepTarget[i+1].append(child.PickTopChild())

                if len(self.StepTarget[i+1]) >= self.Config.BeamWidth:
                    break


        resultNodes = self.StepTarget[len(self.StepTarget)-1]
        resultNodes.sort(key=lambda x:x.Score, reverse=True)

        resultCount = len(resultNodes)

        for i in range(resultCount):

            win = True if i < resultCount/2 else False

            trainData = self.MakeTrainData(resultNodes[i], win)

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

        action = node.GetBestAction_PUCT(self.Config.CPuct, self.Config.DiriclhetAlpha, noiseEnable>0)

        if action == None:
            return None

        value = self.SearchMoves(node.Children[action], noiseEnable-1)

        if value == None:
            return None

        node.N += 1
        node.W += value
        node.Q = node.W / node.N

        return value


    def MakeTrainData(self, node, win):

        assert isinstance(node, Node)

        trainData = list([])
        value = 1 if win else -1

        while True:
            
            node = node.Parent

            if node.Parent == None:
                break

            if node.PickedTransition==None :
                continue

            transition = node.PickedTransition;

            policy = np.power(transition, 1/self.Config.PlicyTau)
            policy /= np.sum(policy)

            trainData.append(list([node.Observation.tolist(), policy.tolist(), value]))

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





