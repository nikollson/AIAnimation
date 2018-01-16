

from Worker.SelfPlay import SelfPlay
from Worker.Optimizer import Optimizer
from Worker.Viewer import Viewer
from Worker.Initializer import Initializer
from Worker.AllConfig import AllConfig
from Worker.Evaluater import Evaluater

allConfig = AllConfig()

evaluater = Evaluater(allConfig)
evaluater.Start()


