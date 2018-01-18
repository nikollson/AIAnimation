

from Worker.SelfPlay import SelfPlay
from Worker.Optimizer import Optimizer
from Worker.Viewer import Viewer
from Worker.Initializer import Initializer
from Worker.AllConfig import AllConfig
from Worker.Evaluater import Evaluater


import time

allConfig = AllConfig()


for _ in range(100000):
    evaluater = Evaluater(allConfig)
    ret = evaluater.Start()

    if ret == True:
        break

    time.sleep(0.2)


