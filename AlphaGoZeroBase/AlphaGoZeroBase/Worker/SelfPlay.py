
from Environment.MujocoEnv  import MujocoEnv as MujocoEnv
from Environment.MujocoModelHumanoid  import MujocoModelHumanoid as Model
from Environment.MujocoTask import MujocoTask, TaskConfig
from Worker.AllConfig import AllConfig
from Network.NetworkModel import NetworkModel as Network
from Agent.Agent import Agent, AgentConfig

import os
import json



class SelfPlay:
    def __init__(self, config:AllConfig):
        self.Config = config

    def Start(self):
        
        filePath = self.Config.FilePath.NextGeneration

        net = Network()
        net.Load(filePath.Config, filePath.Weight)
        
        model = Model()
        task = MujocoTask.Load(model, filePath.Task)
        env = MujocoEnv(model, task)

        agentConfig = self.Config.SelfPlayAgent
        agent = Agent(agentConfig, net, model, task)
        
        state = env.GetSimState()
        bestAction = agent.SearchBestAction(state)

        agent.SaveTrainData(self.Config.GetTrainPath())

