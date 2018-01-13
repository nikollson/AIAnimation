
import json
import os

from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv1D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.losses import mean_squared_error
import keras.backend as K
from keras.optimizers import SGD

class Config:

    Regularizer = l2(1e-4)

    CnnFilterSize = 3
    CnnFilerNum = 30

    MiddleCnnLayerNum = 2
    MiddleCnnFilterNum = 20

    ValueNodeNum = 30
    LearningRate = 1e-2


class NetworkModel:

    def __init__(self, useCPU = True):
        
        self.Model = None

        if useCPU:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


    def Save(self, configPath, weightPath):

        with open(configPath, "wt") as f:
            json.dump(self.Model.get_config(), f)
           
        self.Model.save_weights(weightPath)


    def Load(self, configPath, weightPath):

        with open(configPath, "rt") as f:
            self.Model = Model.from_config(json.load(f))

        self.Model.load_weights(weightPath)


    def Build(self, observationShape, actionNum):

        in_x = x = Input(observationShape)

        # Input Layer

        x = Conv1D(filters=Config.CnnFilerNum, kernel_size=Config.CnnFilterSize,
                   padding="same", kernel_regularizer=Config.Regularizer)(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)


        # Midele Layer

        for _ in range(Config.MiddleCnnLayerNum):

            start_x = x
            
            x = Conv1D(filters=Config.CnnFilerNum, kernel_size=Config.CnnFilterSize, 
                       padding="same", kernel_regularizer=Config.Regularizer)(x)
            x = BatchNormalization(axis=1)(x)
            x = Activation("relu")(x)
            x = Conv1D(filters=Config.CnnFilerNum, kernel_size=Config.CnnFilterSize, 
                       padding="same", kernel_regularizer=Config.Regularizer)(x)
            x = BatchNormalization(axis=1)(x)
            x = Add()([start_x, x])
            x = Activation("relu")(x)

        res_out = x


        # Policy Output Layer

        x = Conv1D(filters=2, kernel_size=1, kernel_regularizer=Config.Regularizer)(res_out)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Flatten()(x)

        policy_out = Dense(actionNum, kernel_regularizer=Config.Regularizer, activation="softmax", name="policy_out")(x)


        # Value Output Layer
        
        x = Conv1D(filters=1, kernel_size=1, kernel_regularizer=Config.Regularizer)(res_out)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        x = Dense(Config.ValueNodeNum, kernel_regularizer=Config.Regularizer, activation="relu")(x)

        value_out = Dense(1, kernel_regularizer=Config.Regularizer, activation="tanh", name="value_out")(x)


        #Build Model

        self.Model = Model(in_x, [policy_out, value_out], name="animation_model")
        

    def OptimizePatch(self, observeArray, policyArray, valueArray):
        self.Model.fit(observeArray, [policyArray, valueArray])

        
    def loss_policy(y_true, y_pred):
        return K.sum(-y_true * K.log(y_pred + K.epsilon()), axis=-1)

    def loss_value(y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    def Compile(self):
        self.optimizer = SGD(lr=Config.LearningRate, momentum=0.9)

        losses = [NetworkModel.loss_policy, NetworkModel.loss_value]
        self.Model.compile(optimizer=self.optimizer, loss=losses)

        