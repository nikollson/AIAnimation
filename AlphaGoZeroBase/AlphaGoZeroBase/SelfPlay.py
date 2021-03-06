
from Worker.SelfPlay import SelfPlay
from Worker.Optimizer import Optimizer
from Worker.Viewer import Viewer
from Worker.Initializer import Initializer
from Worker.AllConfig import AllConfig


allConfig = AllConfig()

initializer = Initializer(allConfig)
initializer.Start()

worker = SelfPlay(allConfig)
worker.Start()

