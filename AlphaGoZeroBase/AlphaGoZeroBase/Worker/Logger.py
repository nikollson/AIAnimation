
import os
import time
from datetime import datetime as dt

class Logger:
    def __init__(self, name):
        
        self.FilePath = "Log_"+name+".txt"

    def AddLog(self, str):

        with open(self.FilePath, "a") as f:
            tstr = ""
            f.write(dt.now().strftime("%y %m %d %H %M %S")+" "+str+"\n")

