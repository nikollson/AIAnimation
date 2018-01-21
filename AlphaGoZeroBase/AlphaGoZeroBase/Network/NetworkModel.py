
import json
import os
import time

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

class BuildConfig:

    def __init__(self, valueSplit):
        self.Regularizer = l2(1e-4)

        self.InputCnnFilterSize = 1
        self.InputCnnFilterNum = 1024
        
        self.MiddleCnnFillterSize = 3
        self.MiddleCnnLayerNum = 4
        self.MiddleCnnFilterNum = 256

        self.ValueNodeNum = 512
        self.ValueSplit = valueSplit

class CompileConfig:

    def __init__(self, learningRate):
        self.LearningRate = learningRate


class NetworkModel:

    def __init__(self, useCPU = True):

        self.Model = None
        self.OptimizeCount = None
        self.TimeLimit = None

        if useCPU:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


    def Save(self, configPath, weightPath):
        
        if os.access(configPath, os.F_OK):
            while os.access(configPath,os.R_OK)==False or os.access(weightPath,os.R_OK)==False:
                time.sleep(0.001)

        with open(configPath, "wt") as f:
            config = self.Model.get_config()
            config["OptimizeCount"] = self.OptimizeCount
            config["TimeLimit"] = self.TimeLimit
            json.dump(config, f)
            self.Model.save_weights(weightPath)


    def Load(self, configPath, weightPath):

        if os.access(configPath, os.F_OK):
            while os.access(configPath,os.W_OK)==False or os.access(weightPath,os.W_OK)==False:
                time.sleep(0.001)
        
        while True:
            try:
                with open(configPath, "rt") as f:
                    config = json.load(f)
                    self.OptimizeCount = config["OptimizeCount"]
                    self.TimeLimit = config["TimeLimit"]
                    self.Model = Model.from_config(config)
                    self.Model.load_weights(weightPath)
                break
            except:
                time.sleep(0.1)


    def Build(self, config:BuildConfig, observationShape, actionN, timeLimit):

        in_x = x = Input(observationShape)

        # Input Layer

        x = Conv1D(filters=config.InputCnnFilterNum, kernel_size=config.InputCnnFilterSize,
                   padding="same", kernel_regularizer=config.Regularizer)(x)
        x = BatchNormalization(axis=1)(x)
        x = Conv1D(filters=config.MiddleCnnFilterNum, kernel_size=config.MiddleCnnFillterSize, 
                    padding="same", kernel_regularizer=config.Regularizer)(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)


        # Midele Layer

        for _ in range(config.MiddleCnnLayerNum):

            start_x = x
            
            x = Conv1D(filters=config.MiddleCnnFilterNum, kernel_size=config.MiddleCnnFillterSize, 
                       padding="same", kernel_regularizer=config.Regularizer)(x)
            x = BatchNormalization(axis=1)(x)
            x = Activation("relu")(x)
            x = Conv1D(filters=config.MiddleCnnFilterNum, kernel_size=config.MiddleCnnFillterSize, 
                       padding="same", kernel_regularizer=config.Regularizer)(x)
            x = BatchNormalization(axis=1)(x)
            x = Add()([start_x, x])
            x = Activation("relu")(x)

        res_out = x


        # Policy Output Layer

        x = Conv1D(filters=2, kernel_size=1, kernel_regularizer=config.Regularizer)(res_out)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Flatten()(x)

        policy_out = Dense(actionN, kernel_regularizer=config.Regularizer, 
                           activation="softmax", name="policy_out")(x)


        # Value Output Layer


        x = Conv1D(filters=1, kernel_size=1, kernel_regularizer=config.Regularizer)(res_out)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        x = Dense(config.ValueNodeNum, kernel_regularizer=config.Regularizer, activation="relu")(x)

        
        value_out = Dense(config.ValueSplit, kernel_regularizer=config.Regularizer, activation="tanh", name="value_out")(x)


        #Build Model

        self.Model = Model(in_x, [policy_out, value_out], name="animation_model")
        self.OptimizeCount = 0
        self.TimeLimit = timeLimit
        

    def OptimizePatch(self, observeArray, policyArray, valueArray):
        self.Model.fit(observeArray, [policyArray, valueArray])

        
    def loss_policy(y_true, y_pred):
        return K.sum(-y_true * K.log(y_pred + K.epsilon()), axis=-1)

    def loss_value(y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    def Compile(self, config:CompileConfig):
        self.optimizer = SGD(lr=config.LearningRate, momentum=0.9)

        losses = [NetworkModel.loss_policy, NetworkModel.loss_value]
        self.Model.compile(optimizer=self.optimizer, loss=losses)

        