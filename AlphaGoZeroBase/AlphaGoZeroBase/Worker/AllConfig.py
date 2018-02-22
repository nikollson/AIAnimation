
from Network.NetworkModel import NetworkModel, BuildConfig, CompileConfig
from Environment.MujocoTask import TaskConfig
from Agent.Agent import AgentConfig, ValueCalcConfig
from datetime import datetime as dt


class ModelFileConfig:
    def __init__(self,name):
        self.Config = name + ".cnf"
        self.Weight = name + ".wgt"

class NetworkConfig:
    def __init__(self):
        self.BestModel = ModelFileConfig("BestModel")
        self.NextGeneration = ModelFileConfig("NextGeneration")


class WorkerConfig:
    def __init__(self):
        
        self.InitialTimeLimit = 0.4

        self.TrainDataMax = 1500
        self.TrainDataRecentlyPar = 0.2
        self.TrainBatchSize = 20000
        self.TrainBatchRecentlyPar = 0.0
        self.TrainLoop = 1
        
        self.CheckPointLength = 50
        self.EvaluateWinRate = 1
        self.EvaluateTimeStepExpand = 1.05
        


class TaskFileConfig:
    def __init__(self):
        self.TrainDir = "TaskTrain"
        self.EvalDir = "TaskEval"

        self.ModelValiation = 5000
        self.TrainNum = 2000
        self.EvalNum = 200


class AllConfig:
    def __init__(self):
        
        self.Worker = WorkerConfig()

        valueCalc = ValueCalcConfig(self.Worker.TrainDataMax)
        self.SelfPlayAgent = AgentConfig(valueCalc, 150, 1, 0, 0, 0.2)
        self.EvaluateAgent = AgentConfig(valueCalc, 4, 1, 0, 0, 0)
        self.ViewerAgent = AgentConfig(valueCalc, 30, 1, 0, 0, 0)
        self.CheckerAgent = AgentConfig(valueCalc, 10, 1, 0, 0, 0.2)

        self.Build = BuildConfig(100)
        self.FilePath = NetworkConfig()
        self.Task = TaskFileConfig()

        self.TrainDir = "Train"
        self.BestLogDir = "BestLog"

    

    def NetworkCompile(self, optimizeStep):

        per = optimizeStep/self.Worker.CheckPointLength

        if per < 0.5:
            return CompileConfig(2e-2)
        if per < 0.7:
            return CompileConfig(5e-3)
    
        return CompileConfig(1e-3)

    def GetBestLog(self):
        return ModelFileConfig(self.BestLogDir + "/BestLog" + self.GetDirStr())

    def GetTrainPath(self, addStr = ""):
        return self.TrainDir + "/Train" + self.GetDirStr() + addStr + ".txt"

    def GetDirStr(self):
        return dt.now().strftime("%y%m%d%H%M%S")


