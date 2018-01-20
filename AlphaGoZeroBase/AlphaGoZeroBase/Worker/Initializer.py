
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
        self.MakeTasks()
        self.MakeGenerationModel()


    def MakeGenerationModel(self):
        
        filePath = self.Config.FilePath

        hasBest = os.path.exists(filePath.BestModel.Config)
        hasBest |= os.path.exists(filePath.BestModel.Weight)
        
        hasNext = os.path.exists(filePath.NextGeneration.Config)
        hasNext |= os.path.exists(filePath.NextGeneration.Weight)

        if hasBest == False:

            model = MujocoModelHumanoid()
            env = MujocoEnv(model)
            
            dataDir = self.Config.Task.TrainDir
            dataList = os.listdir(dataDir)
            task = MujocoTask(model, dataDir+"/"+dataList[0])


            net = NetworkModel()
            net.Build(self.Config.Build, env.GetObservationShape(task), env.GetActionNum(), self.Config.Worker.InitialTimeLimit)

            print("Make best model")

            net.Save(filePath.BestModel.Config, filePath.BestModel.Weight)


        if hasNext == False:
                
            print("Make next generation model from copy of best model")

            shutil.copyfile(filePath.BestModel.Config, filePath.NextGeneration.Config)
            shutil.copyfile(filePath.BestModel.Weight, filePath.NextGeneration.Weight)


    def MakeTasks(self):
        
        config = self.Config.Task

        taskDataList = os.listdir(config.TrainDir)
        evalDataList = os.listdir(config.EvalDir)

        if len(taskDataList)!=0 and len(evalDataList)!=0:
            return

        model = MujocoModelHumanoid()

        model.MakeTaskModelFile(config.ModelValiation, 
                                config.TrainNum, config.TrainDir,
                                config.EvalNum, config.EvalDir)

