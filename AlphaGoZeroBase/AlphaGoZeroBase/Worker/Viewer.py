
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
        
        cnf = "BestLog/BestLog180126082522.cnf"
        wgt = "BestLog/BestLog180126082522.wgt"
        timeLimit = 0.9

        net = Network()
        net.Load(cnf, wgt)
        net.TimeLimit = timeLimit

        model = Model()
        taskName = "TaskEval/EvalTask114.task"
        task = MujocoTask(model, taskName)
        #task = MujocoTask.LoadRandom(model, self.Config.Task.EvalDir)
        env = MujocoEnv(model)

        agentConfig = self.Config.ViewerAgent
        agent = Agent(agentConfig, net, model, task)
        
        bestAction = agent.SearchBestAction()


        while True:

            env.SetSimState(task.StartState)


            for action in bestAction:

                env.Step(action)
                
                #print(env.GetObservation(task))
                env.Render()
