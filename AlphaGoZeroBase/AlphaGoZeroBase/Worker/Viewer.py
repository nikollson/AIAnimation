
from Environment.MujocoEnv  import MujocoEnv as MujocoEnv
from Environment.MujocoModelHumanoid  import MujocoModelHumanoid as Model
from Environment.MujocoTask import MujocoTask, TaskConfig
from Network.NetworkModel import NetworkModel as Network
from Agent.Agent import Agent, AgentConfig

import os
import json

from Worker.AllConfig import AllConfig


class Viewer:
    def __init__(self, config:AllConfig):
        self.Config = config

    def Start(self):
        
        
        filePath = self.Config.FilePath.BestModel

        net = Network()
        net.Load(filePath.Config, filePath.Weight)
        
        model = Model()
        task = MujocoTask.Load(model, filePath.Task)
        env = MujocoEnv(model, task)


        agentConfig = self.Config.ViewerAgent
        agent = Agent(agentConfig, net, model, task)
        
        state = env.GetSimState()
        #bestAction = agent.SearchBestAction(state)

        bestAction = []
        for i in range(80):
            bestAction.append(1)
            bestAction.append(3)
            bestAction.append(1)
            bestAction.append(3)
            bestAction.append(5)
            bestAction.append(1)
            bestAction.append(3)
            bestAction.append(4)
            bestAction.append(4)

        while True:

            env.SetSimState(state)

            for action in bestAction:

                env.Step(action)
                
                print()
                print(env.GetObservation())

                env.Render()

