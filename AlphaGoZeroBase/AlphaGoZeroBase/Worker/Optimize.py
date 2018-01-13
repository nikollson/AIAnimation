
from Network.NetworkModel import NetworkModel

import os
import random
import json
import numpy as np

class Optimize:
    def __init__(self):
        a=3

    def Start(self):

        data = None
        model = None

        print("** Loading **")

        with open("train.txt", "rt") as f:
            data = json.load(f)

        net = NetworkModel()
        net.Load("AA.cnf", "AA.wgt")

        print("** Loaded **")
        
        observeList = []
        policyList = []
        valueList = []

        for p in range(5):
            index = [i for i in range(len(data))]
            random.shuffle(index)

            for i in range(len(index)):
                observeList.append(data[i][0])
                policyList.append(data[i][1])
                valueList.append(data[i][2])

        observeList = np.array(observeList)
        policyList = np.array(policyList)
        valueList = np.array(valueList)

        net.Compile()
        net.OptimizePatch(observeList, policyList, valueList)

        net.Save("AA.cnf", "AA.wgt")
       




