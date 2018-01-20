
from Environment.MujocoModel import MujocoModel
from Environment.MujocoEnv import MujocoEnv
import random
import json
import os

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

    
    def MakeTaskModelFile(self, modelNum, trainNum, taskTrainDir, evalNum, taskEvalDir):
        
        self.MakeHopperTask(modelNum, trainNum, taskTrainDir, evalNum, taskEvalDir)



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

        jointNames = ['torso', 'thigh_joint', 'leg_joint', 'foot_joint', 'toe']

        return jointNames


    def MakeHopperTask(self, modelNum, trainNum, taskTrainDir, evalNum, taskEvalDir):
    
        env = MujocoEnv(self)
        
        state = env.GetSimState()

        jsons = []

        for j in range(modelNum):

            state.qpos[env.Model.MujocoModel.get_joint_qpos_addr("rootx")] = random.uniform(-1.5, 1.5)
            state.qpos[env.Model.MujocoModel.get_joint_qpos_addr("rootz")] = random.uniform(2, 2)
            state.qpos[env.Model.MujocoModel.get_joint_qpos_addr("thigh_joint")] = pt =random.uniform(-0.1, -0.8)
            state.qpos[env.Model.MujocoModel.get_joint_qpos_addr("leg_joint")] = pl = random.uniform(-0.1, -0.8)
            state.qpos[env.Model.MujocoModel.get_joint_qpos_addr("rooty")] = random.uniform(-0.4, 0.4) + (pt + pl)/2
            state.qpos[env.Model.MujocoModel.get_joint_qpos_addr("foot_joint")] = random.uniform(-0.5, 0.5)
            
            env.SetSimState(state)

            ok = False

            for i in range(400):
                env.Step(env.GetActionNum()-1)
                
                sensor = env.GetSensorValue(3, "a_foot_joint")
                if sensor[2] >= 20 and i>=10: 
                    ok=True
                    break;

            if ok==False:
                continue;

            joints = ['thigh_joint','leg_joint','foot_joint','rooty','rootx','rootz']
        
            task = {}
            for i in joints:
                task[i] = env.GetSensorValue(1, "jp_"+i)[0]
        
            jsons.append(task)


        for i in range(trainNum):
           
            filePath = taskTrainDir+"/TrainTask"+str(i)+".task"

            print(filePath)

            task1 = random.choice(jsons)
            task2 = random.choice(jsons)
            
            with open(filePath, "wt") as f:
                json.dump(list([task1, task2]), f)

                
        for i in range(evalNum):
           
            filePath = taskEvalDir+"/EvalTask"+str(i)+".task"

            print(filePath)
            
            task1 = random.choice(jsons)
            task2 = random.choice(jsons)
            
            with open(filePath, "wt") as f:
                json.dump(list([task1, task2]), f)

