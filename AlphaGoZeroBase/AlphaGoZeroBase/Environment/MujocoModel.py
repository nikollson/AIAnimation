
from mujoco_py import load_model_from_path
import numpy as np

class MujocoModel:

    def __init__(self, modelPath : str):
        
        self.MujocoModel = load_model_from_path(modelPath)
        
        self.JointList = self.GetJointList()

        self.NActuator = len(self.JointList)
        self.NAction = self.NActuator * 2 + 1
        
        # self.Naction - 1 means no action
        self.NoneAction = self.NAction - 1

        self.TorqueCofficient = 1



    def GetActionTorque(self, actionNum):

        torque = np.zeros(self.NActuator)

        if actionNum != self.NoneAction:
            dir = (actionNum % 2) * 2 - 1
            torque[int(actionNum/2)] += self.TorqueCofficient * dir
        
        return torque


    def GetJointList(self):
        return []


    class Joint:
        def __init__(self,  joint, site, jointPosition, jointVelocity,
                     accel, velocity, gyro, force, torque):

            self.Joint = joint
            self.Site = site
            self.JointPosition = jointPosition
            self.JointVelocity = jointVelocity
            self.Accel = accel
            self.Velocity = velocity
            self.Gyro = gyro
            self.Force = force
            self.Torque = torque

