
from Network.NetworkModel import NetworkModel, BuildConfig, CompileConfig
from Environment.MujocoTask import TaskConfig
from Agent.Agent import AgentConfig
from datetime import datetime as dt


class ModelFileConfig:
    def __init__(self,name):
        self.Config = name + ".cnf"
        self.Weight = name + ".wgt"
        self.Task = name + ".tsk"

class NetworkConfig:
    def __init__(self):
        self.BestModel = ModelFileConfig("BestModel")
        self.NextGeneration = ModelFileConfig("NextGeneration")


class WorkerConfig:
    def __init__(self):
        self.TrainBatchSize = 2048*5
        self.TrainDataMax = 100
        self.TrainLoop = 10
        self.CheckPointLength = 40
        self.TrainLoop = 5
        self.CheckPointLength = 50
        self.OptimizeReluEdge = 0.2
        self.EvaluateButtle = 1
        self.EvaluateWinRate = 0.49
        self.EvaluateTimeStepSampling = 3000
        self.EvaluateTimeStepUpdateRate = 0.75
        self.EvaluateTimeStepUpdateScale = 1.1

class AllConfig:
    def __init__(self):

        self.SelfPlayAgent = AgentConfig(180, 1, 0.5)
        self.EvaluateAgent = AgentConfig(180, 1, 0)
        self.ViewerAgent = AgentConfig(180, 1, 0)

        self.InitializeTask = TaskConfig(0.5, 0.5)

        self.Build = BuildConfig()
        self.FilePath = NetworkConfig()

        self.TrainDir = "Train"
        self.BestLogDir = "BestLog"

        self.Worker = WorkerConfig()
    

    def NetworkCompile(self, optimizeStep):

        per = optimizeStep/self.Worker.CheckPointLength

        if per < 0.5:
            return CompileConfig(8e-2)

        if per < 0.75:
            return CompileConfig(5e-2)

        return CompileConfig(3e-2)

    def GetBestLog(self):
        return ModelFileConfig(self.BestLogDir + "/BestLog" + self.GetDirStr())

    def GetTrainPath(self, addStr = ""):
        return self.TrainDir + "/Train" + self.GetDirStr() + addStr + ".txt"

    def GetDirStr(self):
        return dt.now().strftime("%y%m%d%H%M%S")


