import numpy as np
from time import time
import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import h5py

import sys
from os import listdir, remove
from os.path import isfile, join, exists, basename, splitext
#from laspy.file import File
from random import randint
from enum import Enum
from math import *
import re

from sys import path

#from notify_run import Notify
#notifyDevice = Notify()

# import firebase_admin
# from firebase_admin import credentials
# from firebase_admin import firestore
# cred = credentials.Certificate('./online-app-600-firebase-adminsdk-zx0e7-bb28b4d148.json')
# firebase_admin.initialize_app(cred)
# firestoreDB = firestore.client()

class FireStroreCollection:
    train = "trainMesages"
    dataProc = "dataProcMesages"

if(os.path.exists("C:/Users/Jonas")):
    mainPath = "G:/PointCloud DataSets/"
elif(os.path.exists("C:/Users/JStanke")):
    mainPath = "E:/PointClouds/"
else:
    mainPath = "/content/drive/My Drive/object detection/"

class Paths:
    datasets = mainPath

    class Semantic3D:
        pointCloudPath = mainPath + "semantic3d/"
    
        rawTrain = pointCloudPath+"rawTrain/"
        rawTest = pointCloudPath+"rawTest/"
        rawTestReduced = pointCloudPath+"rawTestReduced/"
        processedTrain = pointCloudPath+"processedTrain/"
        processedTest = pointCloudPath+"processedTest/"
        processedTestReduced = pointCloudPath+"processedTestReduced/"
        generatedTest = pointCloudPath+"generatedTest/"
        rawSmallPc = rawTrain + "bildstein_station3_xyz_intensity_rgb.hdf5"
        procSmallPc = processedTrain + "bildstein_station3_xyz_intensity_rgb.npy"
    
    class Curbs:
        pointCloudPath = mainPath + "curbs/"
    
        processedTrain = pointCloudPath+"processedTrain/forSegmentation(10cmVoxels)/"
        processedTest = pointCloudPath+"processedTest/forSegmentation(10cmVoxels)/"
        processedTest = pointCloudPath+"processedTest/forSegmentation(10cmVoxels)/"
        generated = pointCloudPath+"/generated/"        

    class NPM3D:
        pointCloudPath = mainPath + "NPM3D/"
        rawTrain = pointCloudPath+"training_10_classes/"
        rawTest = pointCloudPath+"test_10_classes/"
        # processedTrain = pointCloudPath+"processedTrain/"
        processedTrain = pointCloudPath+"torch_generated_data/train_pointclouds/"
        processedTrainVoxels = pointCloudPath+"processedTrainVoxels/"
        processedTest = pointCloudPath+"processedTest/"
        generatedTest = pointCloudPath+"generatedTest/"
    
    class VGTU:
        pointCloudPath = mainPath + "VGTU/"        

    dataPath = "./data/"
    checkPointFilePath = "check.point"
    pausePointFilePath = "pause.point"
    trainLogPath = "./training.log"
    dataProcPath = "./dataProc.log"

    if(not os.path.exists(dataPath)):
        os.mkdir(dataPath)
    	    
    @staticmethod
    def GetFiles(folder, excludeFiles = None, onlyNames = False, withoutExtension = False, findExtesions = ('.hdf5', '.npy', '.las', '.ply')):
        if(isinstance(findExtesions, list)):
            findExtesions = tuple(findExtesions)

        if(excludeFiles is None):
            excludeFiles = []
        if(not isinstance(excludeFiles, list)):
            excludeFiles = [excludeFiles]
        excludeNames = [splitext(basename(name))[0] for name in excludeFiles]

        path = folder + "/"
        if(onlyNames):
            path = ""

        pcFiles = [splitext(basename(path+f))[0] if withoutExtension else path+f 
                for f in listdir(folder)
                if isfile(join(folder, f))
                and f.endswith(findExtesions)
                and not f.startswith('_') 
                and not (splitext(basename(f))[0] in excludeNames)
                and splitext(basename(f))[0] != "small"]

        return pcFiles
    
    @staticmethod
    def JoinPaths(basePath, paths):
        assert(isinstance(paths, list))
        return list(map(lambda path: os.path.join(basePath, path), paths))

    @staticmethod
    def GetBestModel(withPrefix = "0"):
        if(not exists(Paths.dataPath)):
            return None

        modelFiles = [[Paths.dataPath+"/"+f,  float(re.findall("\d+\.\d+", splitext(basename(f))[0])[0])] for f in listdir(Paths.dataPath) if 
                                                                                                            isfile(join(Paths.dataPath, f)) 
                                                                                                            and f.endswith('.h5')
                                                                                                            and basename(f).startswith(withPrefix)]
    
        if(len(modelFiles) == 0):
            return None

        scores = np.array([modelFiles[i][1] for i in range(len(modelFiles))])
        index = np.argmax(scores)

        return modelFiles[index][0]

    @staticmethod
    def FileName(path, withoutExt = True):
        name = os.path.basename(path)
        if(withoutExt):
            name = os.path.splitext(name)[0]
        return name
            
class Label:
    class Semantic3D:
        unlabeled = int(0)
        manMadeTerrain = int(1)
        naturalTerrain = int(2)
        highVegetation = int(3)
        lowVegetation = int(4)
        buildings = int(5)
        hardScape = int(6)
        scanningArtefacts = int(7)
        cars = int(8)
        Count = int(9)
        Names = ["unlabeled", "manMadeTerrain", "naturalTerrain", "highVegetation", "lowVegetation", "buildings", "hardScape", "scanningArtefacts", "cars"]
    
    class Curbs:
        other = int(0)
        curbs = int(1)
        Names = ["other", "curbs"]
        
    class NPM3D:
        unclassified = int(0)
        ground = int(1)
        building = int(2)
        pole_roadSign_trafficLight = int(3)
        bollard_smallPole = int(4)
        trash_can = int(5)
        barrier = int(6)
        pedestrian = int(7)
        car = int(8)
        natural_vegetation = int(9)
        Count = int(10)
        Names = ["unclassified", "ground", "building", "pole", "smallPole", "trash_can", "barrier", "pedestrian", "car", "vegetation"]

class Colors:
    grey = np.array([128, 128, 128])/255
    red = np.array([136, 0, 1])/255
    mint = np.array([170, 255, 195])/255
    teal = np.array([0, 128, 128])/255
    green = np.array([60, 180, 75])/255
    verygreen = np.array([0, 255, 0])/255
    brown = np.array([170, 110, 40])/255
    # white = np.array([255, 255, 255])/255
    black = np.array([0, 0, 0])/255
    blue = np.array([0, 0, 255])/255    
    pink = np.array([255, 56, 152])/255    

    Npm3D = [grey, red, blue, teal, mint, brown, pink, black, green]
    Sema3D = [grey, verygreen, green, mint, red, blue, brown, black]