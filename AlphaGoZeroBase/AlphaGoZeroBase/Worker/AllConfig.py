
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
        self.TrainBatchSize = 20480
        self.TrainDataMax = 100000
        self.CheckPointLength = 5
        self.EvaluateButtle = 7
        self.EvaluateWinRate = 0.55

class AllConfig:
    def __init__(self):

        self.SelfPlayAgent = AgentConfig(5, 4)
        self.ViewerAgent = AgentConfig(5, 4)
        self.EvaluateAgent = AgentConfig(5,4)

        self.InitializeTask = TaskConfig(1, 0.3)

        self.FilePath = NetworkConfig()

        self.TrainDir = "Train"
        self.BestLogDir = "BestLog"

        self.Worker = WorkerConfig()
    

    def NetworkCompile(self):
        return CompileConfig(1e-2)

    def GetBestLog(self):
        return ModelFileConfig(self.BestLogDir + "/BestLog" + self.GetDirStr())

    def GetTrainPath(self):
        return self.TrainDir + "/Train" + self.GetDirStr() + ".txt"

    def GetDirStr(self):
        return dt.now().strftime("%y%m%d%H%M%S")


