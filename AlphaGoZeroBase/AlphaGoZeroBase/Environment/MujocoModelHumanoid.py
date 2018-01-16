
from Environment.MujocoModel import MujocoModel

class MujocoModelHumanoid(MujocoModel):

    def __init__(self):

        #super().__init__("Xml/Tesrt.xml")

        super().__init__("Xml/hopper.xml")

        self.TorqueCofficient = 1


    def GetJointList(self):

        #self.JointNames = self.GetHumanoidJointNames()
        
        self.JointNames = self.GetHopperJointNames()

        jointList = []

        for name in self.JointNames:

            joint = MujocoModel.Joint(
                name, name, 'jp_'+name, 'jv_'+name,
                'a_'+name, 'v_'+name, 'g_'+name, 'f_'+name, 't_'+name)

            jointList.append(joint)

        return jointList

    def GetHumanoidJointNames(self):
        
        jointNames = [
            'abdomen_z', 'abdomen_y', 'abdomen_x', 
            'right_hip_x', 'right_hip_z', 'right_hip_y', 
            'right_knee', 'right_ankle_y', 'right_ankle_x', 
            'left_hip_x', 'left_hip_z', 'left_hip_y', 
            'left_knee', 'left_ankle_y', 'left_ankle_x', 
            'right_shoulder1', 'right_shoulder2', 'right_elbow', 
            'left_shoulder1', 'left_shoulder2', 'left_elbow']

        return jointNames

    def GetHopperJointNames(self):

        jointNames = ['thigh_joint', 'leg_joint', 'foot_joint']

        return jointNames