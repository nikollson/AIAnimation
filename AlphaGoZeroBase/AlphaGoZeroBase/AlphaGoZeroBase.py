

from Worker.SelfPlay import SelfPlay
from Worker.Optimize import Optimize
from Worker.Viewer import Viewer


viewer = Viewer()
viewer.Start()



for step in range(1000):

    print(step)

    worker = SelfPlay()
    worker.Start()

    optimize = Optimize()
    optimize.Start()



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