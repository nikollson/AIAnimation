
from Environment.MujocoModelHumanoid import MujocoModelHumanoid
from Environment.MujocoEnv import MujocoEnv
from Environment.MujocoTask import MujocoTask, TaskConfig
from Network.NetworkModel import NetworkModel, BuildConfig
from Worker.AllConfig import AllConfig

import os
import shutil

class Initializer:
    def __init__(self, config:AllConfig):
        self.Config = config

    def Start(self):

        filePath = self.Config.FilePath

        hasBest = os.path.exists(filePath.BestModel.Config)
        hasBest |= os.path.exists(filePath.BestModel.Weight)
        hasBest |= os.path.exists(filePath.BestModel.Task)
        
        hasNext = os.path.exists(filePath.NextGeneration.Config)
        hasNext |= os.path.exists(filePath.NextGeneration.Weight)
        hasNext |= os.path.exists(filePath.NextGeneration.Task)

        if hasBest == False:

            model = MujocoModelHumanoid()
            env = MujocoEnv(model, MujocoTask(model,self.Config.InitializeTask))

            net = NetworkModel()
            net.Build(BuildConfig(env.GetObservationShape(), env.GetActionNum()))
            
            print("Make best model and minimal task.")

            net.Save(filePath.BestModel.Config, filePath.BestModel.Weight)
            env.Task.Save(filePath.BestModel.Task)


        if hasNext == False:
                
            print("Make next generation model from copy of best model")

            shutil.copyfile(filePath.BestModel.Config, filePath.NextGeneration.Config)
            shutil.copyfile(filePath.BestModel.Weight, filePath.NextGeneration.Weight)
            shutil.copyfile(filePath.BestModel.Task, filePath.NextGeneration.Task)


