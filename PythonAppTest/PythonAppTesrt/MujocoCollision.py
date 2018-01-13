
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
class CapsuleGeometry:
    def __init__(self, position, matrix, radius, width):
        self.position = position
        self.matrix = matrix
        self.radius = radius
        self.width = width

def GetCapsuleCenterLine(a):
    posA = np.dot(a.matrix,[0,0,a.width]) + a.position
    posB = np.dot(a.matrix,[0,0,-a.width]) + a.position
    return posA, posB

def GetCapsuleData(sim, name):
    center = sim.data.get_geom_xpos(name)
    mat = sim.data.get_geom_xmat(name)
    id = sim.model.geom_name2id(name)
    size = sim.model.geom_size[id]
    return CapsuleGeometry(center,mat,size[0],size[1])

def GetProjection_Point2Line(a1,a2,b):
    vec1 = a2-a1
    vec2 = b-a1
    e = vec1 / np.linalg.norm(vec1)
    d = np.dot(vec2,e)
    return d*e + a1

def GetDistance_Line2Point(a1,a2,b):
    p = GetProjection_Point2Line(a1,a2,b)
    if np.dot(p-a1,p-a2) < 0.00001:
        return np.linalg.norm(b-p)
    return min(np.linalg.norm(b-a1), np.linalg.norm(b-a2))

def GetProjection_Point2Plane(a1,a2,a3,b):
    vec1 = a2-a1
    vec2 = a3-a1
    vec2 = vec2 - np.dot(vec1,vec2) * vec1 / pow(np.linalg.norm(vec1),2)
    p1 = GetProjection_Point2Line(a1,a1+vec1,b)
    p2 = GetProjection_Point2Line(a1,a1+vec2,b)
    return p1+p2-a1;

def GetDistance_Line2Line(a1,a2,b1,b2):
    dist1 = GetDistance_Line2Point(a1,a2,b1)
    dist2 = GetDistance_Line2Point(a1,a2,b2)
    dist3 = GetDistance_Line2Point(b1,b2,a1)
    dist4 = GetDistance_Line2Point(b1,b2,a2)
    
    p1 = GetProjection_Point2Plane(a1,a2,a1+b2-b1,b1)
    p2 = GetProjection_Point2Plane(a1,a2,a1+b2-b1,b2)
    pe = b1-p1
    if np.dot(np.cross(a2-a1,p1-a1),pe) * np.dot(np.cross(a2-a1,p2-a1),pe) < 0.00001:
        return np.linalg.norm(pe)

    return min(dist1,dist2,dist3,dist4)

def GetDistance_Capsule2Capsle(a, b):
    a1,a2 = GetCapsuleCenterLine(a)
    b1,b2 = GetCapsuleCenterLine(b)
    return GetDistance_Line2Line(a1,a2,b1,b2)

def GetDistance_Geometry2Geometry(sim, nameA, nameB):
    a = GetCapsuleData(sim,nameA)
    b = GetCapsuleData(sim,nameB)
    return GetDistance_Capsule2Capsle(a,b)

