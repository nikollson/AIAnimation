

from Worker.SelfPlay import SelfPlay
from Worker.Optimizer import Optimizer
from Worker.Viewer import Viewer
from Worker.Initializer import Initializer
from Worker.AllConfig import AllConfig
from Worker.Evaluater import Evaluater

import gc


allConfig = AllConfig()

initializer = Initializer(allConfig)
initializer.Start()

optimizer = Optimizer(allConfig)


for step in range(0):

    print("--- Work " + str(step) + " ---")
    
    #worker = SelfPlay(allConfig)
    #worker.Start()

    #optimizer.Start()

    #evaluater = Evaluater(allConfig)
    #evaluater.Start()
    

viewer = Viewer(allConfig)
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