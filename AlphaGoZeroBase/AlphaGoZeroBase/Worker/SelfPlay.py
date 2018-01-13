
from Environment.MujocoEnv  import MujocoEnv as MujocoEnv
from Environment.MujocoModelSimple  import MujocoModelSimple as Model
from Network.NetworkModel import NetworkModel as Network
from Agent.Agent import Agent, AgentConfig

import os
import json



class SelfPlay:
    def __init__(self):
        a = 3

    def Start(self):
        
        model = Model()
        env = MujocoEnv(model)

        net = Network()
        net.Load("AA.cnf", "AA.wgt")

        agentConfig = AgentConfig(4, 40)
        agent = Agent(agentConfig, net, model)
        
        state = env.GetSimState()
        bestAction = agent.SearchBestAction(state)

        with open("train.txt", "wt") as f:
            json.dump(agent.TrainData, f)
