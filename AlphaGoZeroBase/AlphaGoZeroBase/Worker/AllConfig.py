
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
        self.TrainBatchSize = 1000
        self.TrainDataMax = 2000
        self.CheckPointLength = 1
        self.EvaluateButtle = 1
        self.EvaluateWinRate = 0.49
        self.EvaluateTimeStepSampling = 5000
        self.EvaluateTimeStepUpdateRate = 0.9
        self.EvaluateTimeStepUpdateScale = 1.1

class AllConfig:
    def __init__(self):

        self.SelfPlayAgent = AgentConfig(8, 1, 1, 0.5)
        self.EvaluateAgent = AgentConfig(8,1, 0, 0)
        self.ViewerAgent = AgentConfig(8, 1, 0, 0)

        self.InitializeTask = TaskConfig(0.2, 0.3)

        self.Build = BuildConfig()
        self.FilePath = NetworkConfig()

        self.TrainDir = "Train"
        self.BestLogDir = "BestLog"

        self.Worker = WorkerConfig()
    

    def NetworkCompile(self, optimizeStep):

        per = optimizeStep/self.Worker.CheckPointLength

        if per < 0.4:
            return CompileConfig(1e-2)

        if per < 0.6:
            return CompileConfig(1e-3)

        return CompileConfig(1e-4)

    def GetBestLog(self):
        return ModelFileConfig(self.BestLogDir + "/BestLog" + self.GetDirStr())

    def GetTrainPath(self, addStr = ""):
        return self.TrainDir + "/Train" + self.GetDirStr() + addStr + ".txt"

    def GetDirStr(self):
        return dt.now().strftime("%y%m%d%H%M%S")


