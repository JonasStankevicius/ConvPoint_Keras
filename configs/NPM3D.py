from configs import Config
from imports import *

class NPM3D(Config):
    pointComponents = 3
    featureComponents = 1
    classCount = Label.NPM3D.Count-1
    classNames = Label.NPM3D.Names
    test_step = 0.5
    name = "NPM3D"
    Paths = Paths.NPM3D    

    testFiles = [
                # "Lille1_1_0.npy",
                # "Lille1_1_1.npy",
                # "Lille1_1_2.npy",
                # "Lille1_1_3.npy",
                # "Lille1_1_4.npy",
                # "Lille1_1_5.npy",
                # "Lille1_1_6.npy",
                # "Lille1_1_7.npy",
                # "Lille1_1_8.npy",

                # "Lille1_2_0.npy",
                # "Lille1_2_1.npy",
                
                "Lille2_0.npy",
                "Lille2_1.npy",
                "Lille2_2.npy", 
                "Lille2_8.npy", 
                "Lille2_9.npy",                 

                # "Paris_0.npy",
                # "Paris_1.npy",
                ]
    
    excludeFiles = [
                    # "Lille1_1_7.npy",
                    # "Lille1_2_2.npy",
                    "Lille2_10.npy",
                    # "Paris_2.npy",
                    ]