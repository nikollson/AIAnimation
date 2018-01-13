

from Worker.SelfPlay import SelfPlay
from Worker.Optimizer import Optimizer
from Worker.Viewer import Viewer
from Worker.Initializer import Initializer
from Worker.AllConfig import AllConfig


allConfig = AllConfig()

initializer = Initializer(allConfig)
optimizer = Optimizer(allConfig)
worker = SelfPlay(allConfig)
viewer = Viewer(allConfig)

for step in range(10000):

    print("--- Work " + str(step) + " ---")

    initializer.Start()
    worker.Start()

    optimizer.Start()

viewer.Start()

    


'''

def dump(x):
    print(type(x))
    print(x)
    prev = '*'
    for y in dir(x):
        if prev[0]!=y[0] :
            if prev[0]!='*':
                print()
            print('['+y[0]+']',end=' ')
        print(y,end=', ')
        prev = y
    print()



from mujoco_py import load_model_from_path, MjSim, MjViewer

model = load_model_from_path("Xml/Tesrt.xml")
sim = MjSim(model)


'''