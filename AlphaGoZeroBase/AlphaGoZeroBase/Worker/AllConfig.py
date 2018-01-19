
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
        self.CheckPointLength = 80
        self.TrainDataMax = 80
        self.TrainLoop = 5
        self.OptimizeReluEdge = 0.01
        self.EvaluateButtle = 1
        self.EvaluateWinRate = 0.55
        self.EvaluateTimeStepSampling = 10
        self.EvaluateTimeStepUpdateRate = -1
        self.EvaluateTimeStepUpdateScale = 1.02
        self.EvaluateTimeStepEpsilon = 0.0001

class AllConfig:
    def __init__(self):

        self.SelfPlayAgent = AgentConfig(80, 1, 0.5, 0.3)
        self.EvaluateAgent = AgentConfig(200, 1, 0, 0)
        self.ViewerAgent = AgentConfig(200, 1, 0, 0)

        self.InitializeTask = TaskConfig(0.5, 0.5)

        self.Build = BuildConfig()
        self.FilePath = NetworkConfig()

        self.TrainDir = "Train"
        self.BestLogDir = "BestLog"

        self.Worker = WorkerConfig()
    

    def NetworkCompile(self, optimizeStep):

        per = optimizeStep/self.Worker.CheckPointLength

        if per < 0.6:
            return CompileConfig(3e-2)

        return CompileConfig(5e-3)

    def GetBestLog(self):
        return ModelFileConfig(self.BestLogDir + "/BestLog" + self.GetDirStr())

    def GetTrainPath(self, addStr = ""):
        return self.TrainDir + "/Train" + self.GetDirStr() + addStr + ".txt"

    def GetDirStr(self):
        return dt.now().strftime("%y%m%d%H%M%S")


