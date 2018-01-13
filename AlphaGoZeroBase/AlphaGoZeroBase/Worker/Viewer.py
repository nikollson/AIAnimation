
from Environment.MujocoEnv  import MujocoEnv as MujocoEnv
from Environment.MujocoModelSimple  import MujocoModelSimple as Model
from Network.NetworkModel import NetworkModel as Network
from Agent.Agent import Agent, AgentConfig

import os
import json



class Viewer:
    def __init__(self):
        a = 3

    def Start(self):
        
        model = Model()
        env = MujocoEnv(model)

        net = Network()
        net.Load("AA.cnf", "AA.wgt")

        agentConfig = AgentConfig(10, 10)
        agent = Agent(agentConfig, net, model)
        
        state = env.GetSimState()
        bestAction = agent.SearchBestAction(state)

        while True:

            env.SetSimState(state)

            for action in bestAction:

                env.Step(action)
                env.Render()
