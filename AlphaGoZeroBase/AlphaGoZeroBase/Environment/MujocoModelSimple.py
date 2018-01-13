
from Environment.MujocoModel import MujocoModel

class MujocoModelSimple(MujocoModel):

    def __init__(self):

        super().__init__("Xml/Tesrt.xml")

        self.TorqueCofficient = 1


    def GetJointList(self):

        jointNames = [
            'abdomen_z', 'abdomen_y', 'abdomen_x', 
            'right_hip_x', 'right_hip_z', 'right_hip_y', 
            'right_knee', 'right_ankle_y', 'right_ankle_x', 
            'left_hip_x', 'left_hip_z', 'left_hip_y', 
            'left_knee', 'left_ankle_y', 'left_ankle_x', 
            'right_shoulder1', 'right_shoulder2', 'right_elbow', 
            'left_shoulder1', 'left_shoulder2', 'left_elbow']

        jointList = []

        for name in jointNames:

            joint = MujocoModel.Joint(
                name, name, 'jp_'+name, 'jv_'+name,
                'a_'+name, 'v_'+name, 'g_'+name, 'f_'+name, 't_'+name)

            jointList.append(joint)

        return jointList
