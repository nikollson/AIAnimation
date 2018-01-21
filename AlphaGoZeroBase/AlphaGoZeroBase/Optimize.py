
from Worker.SelfPlay import SelfPlay
from Worker.Optimizer import Optimizer
from Worker.Viewer import Viewer
from Worker.Initializer import Initializer
from Worker.AllConfig import AllConfig
import json
import os

allConfig = AllConfig()

with open(allConfig.FilePath.NextGeneration.Config, "rt") as f:
    config = json.load(f)


if config["OptimizeCount"] <allConfig.Worker.CheckPointLength:
    
    initializer = Initializer(allConfig)
    optimizer = Optimizer(allConfig)

    for _ in range(20):
        ret = optimizer.Start()

        if ret==False:
            break
else:

    initializer = Initializer(allConfig)
    evaluater = Evaluater(allConfig)
    ret = evaluater.Start()
