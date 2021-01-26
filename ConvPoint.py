from dataTool import ReadLabels, ReadXYZ, VisualizePointCloudClassesAsync, modelPath, DataTool
from imports import *
import math
import numpy as np
from time import time    


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Input, BatchNormalization, Dense, Dropout, InputLayer

from sklearn.neighbors import KDTree
from sklearn.metrics import confusion_matrix
from PIL import Image, ImageEnhance, ImageOps

import random
# from notify_run import Notify

class Const:
    @staticmethod
    def IsWindowsMachine():
        if os.path.isdir("C:/Program Files"):
            return True
        else: 
            return False
    
    if os.path.isdir("C:/Program Files"):
        batchSize = 8
    else:
        batchSize = 16 #25

    #Placeholders
    classCount = Label.Semantic3D.Count-1
    classNames = Label.Semantic3D.Names

    testFiles = []
    excludeFiles = []
    Paths = Paths.Semantic3D
    
    epochs = 100
    pointComponents = 3
    featureComponents = 3 #rgb    
    classCount = 0
    npoints = 8192
    blocksize = 8
    test_step = 0.5
    name = ""

    #Algorithm configuration
    noFeature = False
    Fusion = False
    Scale = False
    Rotate = False
    Mirror = False
    Jitter = False
    FtrAugment = False

    logsPath = "./logs"
    ### MODEL CONFIG
    pl = 64
    ### MODEL CONFIG

    def BuildSpecDict(self):
        return {"noFeature" : self.noFeature,
                "Fusion" : self.Fusion,
                "Scale" : self.Scale,
                "Rotate" : self.Rotate,
                "Mirror" : self.Mirror,
                "Jitter" : self.Jitter,
                "FtrAugment" : False if self.noFeature else self.FtrAugment,
                }

    def Name(self, UID = ""):
        modelName = self.name
        
        modelName += f"({len(self.TrainFiles())}&{len(self.TestFiles())})"

        for spec, value in self.BuildSpecDict().items():
            if(value == True):
                modelName += f"({spec})"

        if(UID != ""):
            modelName += f"_{UID}"

        return modelName
    
    @staticmethod
    def RemoveUID(name : str):
        return name.replace(f"_{Const.ParseModelUID(name)}", "")
    
    @staticmethod
    def UID():
        import uuid
        return uuid.uuid4().hex
    
    @staticmethod
    def ParseModelConfig(file):
        config = Paths.FileName(file).split("_")[0].replace("("," ").replace(")","").replace("vox ","").split(" ")

        const = None
        if(config[0] == NPM3D.name):
            const = NPM3D()            
        if(config[0] == Semantic3D.name):
            const = Semantic3D()
        
        for conf in config[1:]:
            if conf == "noFeature" or conf == "NOCOL":
                const.noFeature = True
            elif conf == "Fusion":
                const.Fusion = True
            elif conf == "Scale":
                const.Scale = True
            elif conf == "Rotate":
                const.Rotate = True
            elif conf == "Mirror":
                const.Mirror = True
            elif conf == "Jitter":
                const.Jitter = True
            elif conf == "FtrAugment":
                const.FtrAugment = True
         
        return const
    
    @staticmethod
    def ParseModelUID(file):
        parts = Paths.FileName(file).split("_")

        if(len(parts) >= 2):
            return parts[1]
        else:
            return None

    @staticmethod
    def ParseModelName(file, withUID = True):
        parts = Paths.FileName(file, withoutExt = False).split("_")

        name = parts[0]
        if(withUID and len(parts) > 1):
            name += "_"+parts[1]

        return name

    def TestFiles(self):        
        return Paths.JoinPaths(self.Paths.processedTrain, self.testFiles)

    def TrainFiles(self):
        return Paths.GetFiles(self.Paths.processedTrain, excludeFiles = self.TestFiles()+self.excludeFiles)

class Semantic3D(Const):        
    pointComponents = 3
    featureComponents = 3 #rgb
    classCount = Label.Semantic3D.Count-1
    classNames = Label.Semantic3D.Names
    test_step = 0.8
    name = "Sem3D"
    Paths = Paths.Semantic3D

    testFiles = [
                "untermaederbrunnen_station3_xyz_intensity_rgb_voxels.npy",
                "domfountain_station1_xyz_intensity_rgb_voxels.npy",
                ]

    excludeFiles = []

    fileNames = {"birdfountain_station1_xyz_intensity_rgb" : "birdfountain1",
                "castleblatten_station1_intensity_rgb" : "castleblatten1",
                "castleblatten_station5_xyz_intensity_rgb" : "castleblatten5",
                "marketplacefeldkirch_station1_intensity_rgb" : "marketsquarefeldkirch1",
                "marketplacefeldkirch_station4_intensity_rgb" : "marketsquarefeldkirch4",
                "marketplacefeldkirch_station7_intensity_rgb" : "marketsquarefeldkirch7",
                "sg27_station3_intensity_rgb" : "sg27_3",
                "sg27_station6_intensity_rgb" : "sg27_6",
                "sg27_station8_intensity_rgb" : "sg27_8",
                "sg27_station10_intensity_rgb" : "sg27_10",
                "sg28_station2_intensity_rgb" : "sg28_2",
                "sg28_station5_xyz_intensity_rgb" : "sg28_5",
                "stgallencathedral_station1_intensity_rgb" : "stgallencathedral1",
                "stgallencathedral_station3_intensity_rgb" : "stgallencathedral3",
                "stgallencathedral_station6_intensity_rgb" : "stgallencathedral6",

                "MarketplaceFeldkirch_Station4_rgb_intensity-reduced" : "marketsquarefeldkirch4-reduced",
                "sg27_station10_rgb_intensity-reduced" : "sg27_10-reduced",
                "sg28_Station2_rgb_intensity-reduced" : "sg28_2-reduced",
                "StGallenCathedral_station6_rgb_intensity-reduced" : "stgallencathedral6-reduced",
                }

class Curbs(Const):        
    pointComponents = 3
    featureComponents = 3
    classCount = 2
    classNames = Label.Curbs.Names
    test_step = 0.5
    name = "Curbs"
    Paths = Paths.Curbs

    if os.path.isdir("C:/Program Files"):
        batchSize = 8
    else:
        batchSize = 25

    testFiles = [
                    "park_extracted.npy",
                    "WCO MX9 extracted.npy",
                    "Copy of S2222791_20181016-105709_0001-002_extracted_part2.npy",
                ]
    
    excludeFiles = [
                    "powerlines_dataset"
                ]

    def FilterCurbAndLineFiles(self, files):
        return [file for file in files if not file.endswith("_curbs.npy") and not file.endswith("_lines.npy")]

    def TestFiles(self):        
        return self.FilterCurbAndLineFiles(super(Curbs, self).TestFiles())

    def TrainFiles(self):
        return self.FilterCurbAndLineFiles(super(Curbs, self).TrainFiles())

class NPM3D(Const):
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

class WeightsMul(tf.keras.layers.Layer):
    def __init__(self, shape, lowBound, highBound, **kwargs):
        super(WeightsMul, self).__init__(**kwargs)
        self.shape = shape
        self.lowBound = lowBound
        self.highBound = highBound

    def build(self, input_shape):
        init = tf.random_uniform_initializer(self.lowBound, self.highBound)
        self.vars = self.add_weight(shape=(self.shape), 
                                    initializer = init, 
                                    trainable = True, dtype=tf.float32)

    def call(self, inputs):        
        return tf.matmul(inputs, self.vars)
    
    def get_config(self):
        config = super(WeightsMul, self).get_config()
        config.update({'shape': self.shape, 'lowBound': self.lowBound, 'highBound': self.highBound})
        return config

class GatherNDLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):        
        super(GatherNDLayer, self).__init__(**kwargs)
    
    def call(self, array, indices):
        return tf.gather_nd(array, indices, batch_dims=1)
    
    def get_config(self):
        config = super(GatherNDLayer, self).get_config()
        return config

class SubstractCenters(tf.keras.layers.Layer):
    def __init__(self, dim, n_centers, **kwargs):
        super(SubstractCenters, self).__init__(**kwargs)
        self.dim = dim
        self.n_centers = n_centers
    
    def build(self, input_shape):
        center_data = np.zeros((self.dim, self.n_centers))
        for i in range(self.n_centers):
            coord = np.random.rand(self.dim)*2 - 1
            while (coord**2).sum() > 1:
                coord = np.random.rand(self.dim)*2 - 1
            center_data[:,i] = coord

        self.centers = self.add_weight(shape = (center_data.shape), 
                                        initializer = tf.constant_initializer(center_data), 
                                        trainable = True, dtype=tf.float32)

    def call(self, points):        
        return points - self.centers
    
    def get_config(self):
        config = super(SubstractCenters, self).get_config()
        config.update({'dim': self.dim, 'n_centers': self.n_centers})
        return config

class UnitBallNormalize(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(UnitBallNormalize, self).__init__(**kwargs)

    def call(self, points):
        maxi = tf.sqrt(tf.reduce_max(tf.reduce_sum(tf.square(tf.stop_gradient(points)), axis = 3), axis = 2))
        maxi = tf.where(tf.equal(maxi, 0.0), tf.constant(1.0), maxi)
        points = points / tf.expand_dims(tf.expand_dims(maxi, 2), 3)
        return points
    
    def get_config(self):
        config = super(UnitBallNormalize, self).get_config()
        return config

def PtConv(fts, points, K, next_pts, in_features, out_features, n_centers = 16):
    next_pts_ = None
    if isinstance(next_pts, int) and points.get_shape()[1] != next_pts:
        # convolution with reduction
        indices, next_pts_ = KDTreeSampleLayer(K, next_pts)(points)
    elif (next_pts is None) or (isinstance(next_pts, int) and points.get_shape()[1] == next_pts):
        # convolution without reduction
        indices = KDTreeLayer(K)(points, points)
        next_pts_ = points
    else:
        # convolution with up sampling or projection on given points
        indices = KDTreeLayer(K)(points, next_pts)
        next_pts_ = next_pts
    
    if next_pts is None or isinstance(next_pts, int):
        next_pts = next_pts_

    # get the features and point cooridnates associated with the indices
    pts = GatherNDLayer()(points, indices)
    if fts is None:
        features = tf.expand_dims(tf.ones_like(pts[:,:,:,0]), 3)
    else:
        features = GatherNDLayer()(fts, indices)        

    # center the neighborhoods
    pts = pts - tf.expand_dims(next_pts,2)

    # normalize to unit ball, or not
    pts = UnitBallNormalize()(pts)

    # compute the distances
    dists = SubstractCenters(3, n_centers)(tf.expand_dims(pts, 4))

    dShape = dists.shape
    dists = tf.reshape(dists, (-1, dShape[1], dShape[2], dShape[3]*dShape[4]))

    dists = DenseInitialized(2*n_centers, activation="relu")(dists)
    dists = DenseInitialized(n_centers, activation="relu")(dists)
    dists = DenseInitialized(n_centers, activation="relu")(dists)
    
    # compute features    
    fs = features.shape # [batch, points, n_centers, in_features]
    ds = dists.shape

    features = tf.transpose(features,[0, 1, 3, 2])
    features = tf.reshape(features, (-1, features.shape[2], features.shape[3])) #features.shape[0]*features.shape[1]
    dists = tf.reshape(dists, (-1, dists.shape[2], dists.shape[3])) #dists.shape[0]*dists.shape[1]

    features = tf.matmul(features, dists)
    features = tf.reshape(features, (-1, ds[1], features.shape[1]*features.shape[2]))

    bound = math.sqrt(3.0) * math.sqrt(2.0 / (in_features + out_features))
    features = WeightsMul([in_features * n_centers, out_features], -bound, bound)(features)

    features = features / fs[2]

    # normalization and activation
    features = BatchNormalization(epsilon = 1e-05, momentum=0.9)(features)    

    features = tf.nn.relu(features)

    return features, next_pts

def LinearInitializer(k):
    k = np.sqrt(1.0/float(k))
    return tf.random_uniform_initializer(k*-1, k)

def DenseInitialized(out_features, activation = None, name = None):
    def DenseInit(x):
        return Dense(out_features, 
                    kernel_initializer = tf.initializers.lecun_normal(),
                    bias_initializer = tf.initializers.lecun_normal(),
                    activation = activation,
                    name = name,
                    )(x)

    return DenseInit

def CreateModel(classCount, ftsComp, in_fts = None, in_pts = None, returnFeatures = False, noColor = False, applySoftmax = True):
    print("Creating new model...")
    
    if(in_fts is None and in_pts is None):
        in_pts = Input(shape=(Const.npoints, Const.pointComponents), dtype=tf.float32) #points 

        if(noColor):
            in_fts = None
        else:
            in_fts = Input(shape=(Const.npoints, ftsComp), dtype=tf.float32) #featuress        
    
    if(noColor):
        in_fts = None

    pl = Const.pl
    ### Down Sample
    x0, _    = PtConv(in_fts, in_pts,   K = 16, next_pts = None,    in_features = ftsComp,  out_features = pl)
    x1, pts1 = PtConv(x0, in_pts,       K = 16, next_pts = 2048,    in_features = pl,       out_features = pl)
    x2, pts2 = PtConv(x1, pts1,         K = 16, next_pts = 1024,    in_features = pl,       out_features = pl)
    x3, pts3 = PtConv(x2, pts2,         K = 16, next_pts = 256,     in_features = pl,       out_features = pl)
    x4, pts4 = PtConv(x3, pts3,         K = 8,  next_pts = 64,      in_features = pl,       out_features = pl*2)
    x5, pts5 = PtConv(x4, pts4,         K = 8,  next_pts = 16,      in_features = pl*2,     out_features = pl*2)
    x6, pts6 = PtConv(x5, pts5,         K = 4,  next_pts = 8,       in_features = pl*2,     out_features = pl*2)

    ## Up Sample
    x5d, _ = PtConv(x6, pts6, K = 4, next_pts = pts5, in_features = pl*2, out_features = pl*2)
    x5d = tf.concat([x5d, x5], axis = 2)

    x4d, _ = PtConv(x5d, pts5, K = 4, next_pts = pts4, in_features = pl*4, out_features = pl*2)
    x4d = tf.concat([x4d, x4], axis = 2)

    x3d, _ = PtConv(x4d, pts4, K = 4, next_pts = pts3, in_features = pl*4, out_features = pl)
    x3d = tf.concat([x3d, x3], axis = 2)

    x2d, _ = PtConv(x3d, pts3, K = 8, next_pts = pts2, in_features = pl*2, out_features = pl)
    x2d = tf.concat([x2d, x2], axis = 2)

    x1d, _ = PtConv(x2d, pts2, K = 8, next_pts = pts1, in_features = pl*2, out_features = pl)
    x1d = tf.concat([x1d, x1], axis = 2)

    x0d, _ = PtConv(x1d, pts1, K = 8, next_pts = in_pts, in_features = pl*2, out_features = pl)
    x0d = tf.concat([x0d, x0], axis = 2)
  
    ### Output layer
    out_labels = Dropout(rate=0.5)(x0d)
    
    out_labels = tf.reshape(out_labels, (-1, out_labels.shape[2]))
    
    out_labels = DenseInitialized(classCount)(out_labels)

    out_labels = tf.reshape(out_labels, (-1, x0d.shape[1], out_labels.shape[1]))

    if(applySoftmax):
        out_labels = tf.nn.softmax(out_labels)

    if(noColor):
        inputList = [in_pts]
    else:
        inputList = [in_fts, in_pts]

    if(returnFeatures):
        return Model(inputList, [x0d, out_labels], name ="model")
        
    model = Model(inputList, out_labels, name ="model")
    model = CompileModel(model, classCount)        
    # print(model.summary())
    return model

def ModifyModelOutput(model, classCount):
    dropoutLayer = model.layers[len(model.layers)-5] #take output of the drop out layer
    out_labels = dropoutLayer.output

    out_labels = tf.reshape(out_labels, (-1, out_labels.shape[2]), name = "lbl_reshape_1")
    out_labels = DenseInitialized(classCount, name = "lbl_dense")(out_labels)    
    out_labels = tf.reshape(out_labels, (-1, dropoutLayer.input.shape[1], out_labels.shape[1]), name = "lbl_reshape_2")
    out_labels = tf.nn.softmax(out_labels, name = "lbl_softmax")

    return Model(model.inputs, out_labels, name ="model")

def ReadModel(modelPath):
    if(not modelPath.endswith(".h5")):
        modelPath += ".h5"

    if(not os.path.exists(modelPath)):
        if(os.path.exists(os.path.join("." , "data", modelPath))):
            modelPath = os.path.join("." , "data", modelPath)
        elif(os.path.exists(os.path.join("." , "data", Const.ParseModelName(modelPath, False)))):
            file = os.path.basename(modelPath)
            folder = os.path.join("." , "data", Const.ParseModelName(modelPath, False))
            modelPath = os.path.join(folder, file)
        elif(os.path.exists(os.path.join("." , "data", Const.ParseModelName(modelPath)))):
            file = os.path.basename(modelPath)
            folder = os.path.join("." , "data", Const.ParseModelName(modelPath))
            modelPath = os.path.join(folder, file)

        if(not os.path.exists(modelPath)):
            raise FileNotFoundError    

    model = tf.keras.models.load_model(modelPath, compile=False,
        custom_objects={'NearestNeighborsLayer': NearestNeighborsLayer, 
                        'SampleNearestNeighborsLayer': SampleNearestNeighborsLayer,
                        'SubstractCenters': SubstractCenters,
                        'WeightsMul': WeightsMul,
                        'GatherNDLayer':GatherNDLayer,
                        'UnitBallNormalize':UnitBallNormalize,
                        'KDTreeSampleLayer':KDTreeSampleLayer,
                        'KDTreeLayer':KDTreeLayer,
                        })

    PrintToLog("{} model loaded".format(modelPath))
    return model

def LatestModel(path):
    if(Const.ParseModelUID(path) is None):
        folders = [os.path.join("." , "data",folder) for folder in os.listdir(os.path.join("." , "data")) 
                                                        if os.path.isdir(os.path.join("." , "data",folder)) 
                                                        and path == Const.RemoveUID(Const.ParseModelName(folder))
                                                        and len(Paths.GetFiles(os.path.join("." , "data",folder), findExtesions=".h5")) > 0]
        path = max(folders, key=os.path.getctime)
    else:
        path = os.path.join("." , "data", Const.ParseModelName(path))    

    try:
        latestModel = max(Paths.GetFiles(path, findExtesions=".h5"), key=os.path.getctime)
    except:
        print(f"No model found in: {path}")
        latestModel = None

    return latestModel

import re
def ModelValMIOU(path):
    result = re.findall("val\((.+)\)", path)
    return float(result[0])

def HighestValMIOUModel(path):
    if(not os.path.isdir(path)):
        path = os.path.join("." , "data", os.path.basename(path).split("_")[0])

    latestModel = max(Paths.GetFiles(path, findExtesions=".h5"), key=ModelValMIOU)
    return latestModel

def LoadModel(modelPath, consts):
    model = ReadModel(modelPath)

    modified = False
    if(model.output.shape[2] != consts.classCount):
        print("Model output {} classes changed to {}".format(model.output.shape[2], consts.classCount))
        modified = True
        model = ModifyModelOutput(model, consts.classCount)

    model = CompileModel(model, consts.classCount)
    # model.summary()
    return model, modified

def ReadModelConfig(path):
    Model = ReadModel(path)
    modelConfig = Const.ParseModelConfig(path)
    return Model, modelConfig

def CreateModelCopy(Model, modelConfig, in_pts, in_RGB):
    inputFeatures = 1 if modelConfig.noFeature else modelConfig.featureComponents
    newModel = CreateModel(modelConfig.classCount, inputFeatures, in_RGB, in_pts, noColor=modelConfig.noFeature, returnFeatures=True, applySoftmax=False)

    if(Model != None):
        for new_layer, layer in zip(newModel.layers, Model.layers):
            new_layer.set_weights(layer.get_weights())

    return newModel

def FuseModels(modelPaths, consts):
    fusionModel = None

    assert(len(modelPaths) == 2 or modelPaths is None)
    print("Model fusion")
    
    if(not modelPaths is None):
        ModelA, modelAConfig = ReadModelConfig(modelPaths[0])
        ModelB, modelBConfig = ReadModelConfig(modelPaths[1])
    else:
        consts.noFeature = False
        modelAConfig = consts
        consts.noFeature = True
        modelBConfig = consts

    in_RGB = None
    if(not modelAConfig.noFeature or not modelBConfig.noFeature):
        in_RGB = Input(shape=(Const.npoints, consts.featureComponents), dtype=tf.float32, name = "In_RGB") #features
    in_pts = Input(shape=(Const.npoints, Const.pointComponents), dtype=tf.float32, name = "In_pts") #points

    newModelA = CreateModelCopy(ModelA, modelAConfig, in_pts, in_RGB)
    newModelB = CreateModelCopy(ModelB, modelBConfig, in_pts, in_RGB)

    x = tf.concat((newModelA.output[0], newModelB.output[0]), axis = 2) #fuse features from both models

    x1, _    = PtConv(x, in_pts,   K = 16, next_pts = Const.npoints,    in_features = 2*128,  out_features = 96)
    x2, _    = PtConv(x1, in_pts,   K = 16, next_pts = Const.npoints,    in_features = 96,  out_features = 48)
    x0d = tf.concat([x2, newModelA.output[1], newModelB.output[1]], axis = 2)

    out_labels = tf.reshape(x0d, (-1, x0d.shape[2]))
    out_labels = Dropout(rate=0.5)(out_labels)
    out_labels = DenseInitialized(consts.classCount)(out_labels)
    out_labels = tf.reshape(out_labels, (-1, x0d.shape[1], out_labels.shape[1]))

    out_labels = tf.nn.softmax(out_labels)

    fusionModel = Model([in_pts] if in_RGB is None else [in_RGB, in_pts], out_labels, name ="model")

    nontrainableNames = [x.name for x in newModelA.layers] + [x.name for x in newModelB.layers]
    # nontrainableNames = [x.name for x in newModelA.layers]
    count = 0
    for i, layer in enumerate(fusionModel.layers):
        if(layer.name in nontrainableNames):
            layer.trainable = False
            count += 1

    PrintToLog(f"{len(fusionModel.layers)-count}/{len(fusionModel.layers)} layers are trainable.")

    fusionModel = CompileModel(fusionModel, consts.classCount)
    # fusionModel.summary()
    return fusionModel

class MIOU(tf.keras.metrics.Metric):
    
    def __init__(self, classCount, name='miou', **kwargs):
        super(MIOU, self).__init__(name=name, **kwargs)
        self.cm = self.add_weight(name=name, shape = (classCount, classCount), initializer='zeros', dtype = tf.int64)
        self.classCount = classCount

    def update_state(self, y_true, y_pred, sample_weight=None):
        TrueLbl = tf.argmax(tf.reshape(y_true, [-1, self.classCount]), axis= 1)
        PredLbl = tf.argmax(tf.reshape(y_pred, [-1, self.classCount]), axis= 1)
        confusion_matrix = tf.math.confusion_matrix(TrueLbl, PredLbl, self.classCount)          
        self.cm.assign_add(tf.cast(confusion_matrix, tf.int64))

    def result(self):
        union = tf.linalg.diag_part(self.cm)
        rowSum = tf.math.reduce_sum(self.cm, axis = 0)
        colSum = tf.math.reduce_sum(self.cm, axis = 1)
        intersection = (colSum + rowSum - union)
        intersection = tf.where(tf.equal(intersection, tf.constant(0, dtype=tf.int64)), tf.constant(1, dtype=tf.int64), intersection)
        iou =  union / intersection
        miou = tf.expand_dims(tf.convert_to_tensor(tf.reduce_sum(iou) / tf.cast(iou.shape[0], dtype=np.float64)), 0)
        return tf.concat((tf.expand_dims(miou,1), tf.cast(tf.expand_dims(iou,1), tf.float64)), 0)

    def reset_states(self):
      # The state of the metric will be reset at the start of each epoch.
      self.cm.assign(tf.zeros((self.classCount, self.classCount), dtype=tf.int64))

def moving_miou_metric(classCount):
    def moving_iou(y_true, y_pred):
        TrueLbl = tf.argmax(tf.reshape(y_true, [-1, classCount]), axis= 1)
        PredLbl = tf.argmax(tf.reshape(y_pred, [-1, classCount]), axis= 1)

        cm = tf.math.confusion_matrix(TrueLbl, PredLbl, classCount)

        union = tf.linalg.diag_part(cm)

        rowSum = tf.math.reduce_sum(cm, axis = 0)
        colSum = tf.math.reduce_sum(cm, axis = 1)

        intersection = (colSum + rowSum - union)+1

        iou =  union / intersection

        return tf.reduce_sum(iou) / tf.cast(tf.math.maximum(iou.shape[0], 1), dtype=np.float64)

    return moving_iou

class IOU(tf.keras.metrics.Metric):
    def __init__(self, classCount, classIndex, name='iou', **kwargs):
        super(IOU, self).__init__(name=name, **kwargs)
        self.cm = self.add_weight(name=name, shape = (classCount, classCount), initializer='zeros', dtype = tf.int64)
        self.classCount = classCount
        self.classIndex = classIndex

    def update_state(self, y_true, y_pred, sample_weight=None):
        TrueLbl = tf.argmax(tf.reshape(y_true, [-1, self.classCount]), axis= 1)
        PredLbl = tf.argmax(tf.reshape(y_pred, [-1, self.classCount]), axis= 1)
        confusion_matrix = tf.math.confusion_matrix(TrueLbl, PredLbl, self.classCount)
        self.cm.assign_add(tf.cast(confusion_matrix, tf.int64))

    def result(self):
        union = tf.linalg.diag_part(self.cm)
        rowSum = tf.math.reduce_sum(self.cm, axis = 0)
        colSum = tf.math.reduce_sum(self.cm, axis = 1)
        intersection = (colSum + rowSum - union)
        intersection = tf.where(tf.equal(intersection, tf.constant(0, dtype=tf.int64)), tf.constant(1, dtype=tf.int64), intersection)
        iou =  union / intersection
        return tf.cast(tf.expand_dims(iou, 1)[self.classIndex], tf.float64)

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.cm.assign(tf.zeros((self.classCount, self.classCount), dtype=tf.int64))

def weighted_categorical_crossentropy(weights):
    # weights = [0.9,0.05,0.04,0.01]
    def wcce(y_true, y_pred):
        Kweights = tf.constant(weights)
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred) * tf.math.reduce_sum(y_true * Kweights, axis=-1)

    return wcce

def CompileModel(model, classCount):
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, epsilon = 1e-8),
        loss = tf.keras.losses.CategoricalCrossentropy(),
        # loss = weighted_categorical_crossentropy([0.7, 5]),
        metrics= [IOU(classCount, 0, name="other"), IOU(classCount, 1, name="curb")] if classCount == 2 else [MIOU(classCount)]
    )
    return model

class IOUPerClass(tf.keras.callbacks.Callback):
    def __init__(self, plot_path, classNames, firstEpoch = 0, metric = "miou"):
        self.metric = metric
        self.epoch = firstEpoch    
        self.classCount = len(classNames)
        self.classNames = classNames
        self.path = plot_path

        print(f"IOU logs path: {self.path}")

        self.writers = []
        self.val_writers = []
        ioupath = os.path.join(plot_path, "iou")
        os.makedirs(ioupath, exist_ok=True)
        for i in range(self.classCount):
            path = os.path.join(ioupath, classNames[i])
            os.makedirs(path, exist_ok=True)
            self.writers.append(tf.summary.create_file_writer(path))

            path = os.path.join(ioupath, "val_"+classNames[i])
            os.makedirs(path, exist_ok=True)
            self.val_writers.append(tf.summary.create_file_writer(path))
            # print("Writer path: ", path)
        
        self.InitializeMIOUWriter()        

    def InitializeMIOUWriter(self):
        mioupath = os.path.join(self.path, "miou")
        os.makedirs(mioupath, exist_ok=True)

        path = os.path.join(mioupath, "miou")
        os.makedirs(path, exist_ok=True)
        self.miou_writer = tf.summary.create_file_writer(path)

        path = os.path.join(mioupath, "val_miou")
        os.makedirs(path, exist_ok=True)
        self.val_miou_writer = tf.summary.create_file_writer(path)
    
    def WriteLog(self, writer, metric, logs, epoch, tag = "miou"):
        value = logs.get(metric)
        if(value is None):
            print(f"Failed getting {metric} log")
            return False
        
        with writer.as_default():
            tf.summary.scalar(tag, value[0][0], step=epoch)
            writer.flush()

    def WriteLogs(self, writers, metric, logs, epoch, tag = "iou"):
        metrix = logs.get(metric)
        if(metrix is None):
            print(f"Failed getting {metric} log")
            return False

        iou = [i[0] for i in metrix[len(metrix)-self.classCount:]]
        for i in range(len(iou)):
            with writers[i].as_default():
                tf.summary.scalar(tag, iou[i], step=epoch)
            writers[i].flush()
    
    def on_epoch_end(self, batch, logs=None):
        self.WriteLogs(self.writers, self.metric, logs, self.epoch)
        self.WriteLogs(self.val_writers, "val_"+self.metric, logs, self.epoch)

        self.WriteLog(self.miou_writer, self.metric, logs, self.epoch)
        self.WriteLog(self.val_miou_writer, "val_"+self.metric, logs, self.epoch)
        self.epoch += 1

logSaveDir = ""
def WriteToLog(msg):
    if(os.path.isdir(logSaveDir)):
        logFile = open(logSaveDir+f"/training.log", "a")
        logFile.write(msg+"\n")
        logFile.close()

def PrintToLog(msg):
    print(msg)
    WriteToLog(msg)

class ModelSaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, saveDir, trainingSteps, metric = "accuracy", modelNamePrefix = "", sendNotifications = False):
        super().__init__()
        self.saveDir = saveDir
        self.metric = metric
        self.modelNamePrefix = modelNamePrefix

        self.epoch = 0
        self.trainingSteps = trainingSteps
        
        self.sendNotifications = sendNotifications
        if(self.sendNotifications):
            self.notifyDevice = Notify()
        
        os.makedirs(self.saveDir, exist_ok=True)
        WriteToLog(f"Training: {modelNamePrefix}")

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch + 1
        if(len(logs) > 0):
            miou = logs.get(self.metric)[0]*100
            val_metric = "val_"+self.metric
            val_miou = logs.get(val_metric)[0]*100
            SaveModel(self.saveDir, epoch, self.model, miou, val_miou, self.modelNamePrefix)

            message = "Ep: {0}. {1}: {2:.3}%. {3}: {4:.3}%".format(self.epoch, self.metric, miou, val_metric, val_miou)
            WriteToLog(message)

            if(self.sendNotifications):
                try:                    
                    self.notifyDevice.send(self.modelNamePrefix + " " + message)
                except:
                    print("notifyDevice error")
    
    # def on_batch_end(self, batch, logs=None):
    #     progress = batch/self.trainingSteps * 100
    #     if(progress % 10 == 0):
    #         try:
    #             message = "Ep. {0} {1}% done. {2}: {3:.3}%".format(self.epoch+1, int(progress), self.metric, logs.get(self.metric)*100)
    #             self.notifyDevice.send(message)
    #         except:
    #             print("notifyDevice error")

def ParseEpoch(modelPath):
    filename = os.path.basename(modelPath)
    return int(filename.split("_")[2])

def GetValidationData(testFiles, consts, batchesCount = 100, newDataGeneration = False):
    print("Gathering validation data...")
    print(f"Test files: {testFiles}")

    if(newDataGeneration):
        PrintToLog("Use TestSequence for validation.")

        assert(len(testFiles) == 1)
        seq = TestSequence(testFiles[0], consts, test = True)
    else:
        PrintToLog("Use TrainSequence for validation.")

        seq = TrainSequence(testFiles, batchesCount, consts, dataAugmentation = False)    
    
    if not consts.noFeature:
        ftsList = np.zeros((0, consts.npoints, consts.featureComponents), np.float32)
    ptsList = np.zeros((0, consts.npoints, 3), np.float32)
    lbsList = np.zeros((0, consts.npoints, consts.classCount), np.uint8)

    if(newDataGeneration):
        indexes = np.arange(min(batchesCount, len(seq)))
        np.random.shuffle(indexes)
    else:
        indexes = range(batchesCount)

    for i in indexes:
        if consts.noFeature:
            if(newDataGeneration):
                ptslbl = seq.__getitem__(i)
            else:
                pts, lbl = seq.__getitem__(i)
                ptslbl = [pts[0], lbl]
            
            ptsList = np.concatenate((ptsList, ptslbl[0]), 0)
            lbsList = np.concatenate((lbsList, ptslbl[1]), 0)
        else:
            if(newDataGeneration):
                ftsptslbl = seq.__getitem__(i)
            else:
                ftspts, lbl = seq.__getitem__(i)
                ftsptslbl = [ftspts[0], ftspts[1], lbl]
            
            ftsList = np.concatenate((ftsList, ftsptslbl[0]), 0)
            ptsList = np.concatenate((ptsList, ftsptslbl[1]), 0)
            lbsList = np.concatenate((lbsList, ftsptslbl[2]), 0)
    
    PrintToLog(f"Generated {len(lbsList)} validation samples.")

    if consts.noFeature:
        return (ptsList, lbsList)
    else:
        return ([ftsList, ptsList], lbsList)
        
def TrainModel(trainFiles, testFiles, consts : Const, modelPath = None, saveDir = Paths.dataPath, classes = None, first_epoch = 0, epochs = None, sendNotifications = False):    
    model = None
    modelName = None
    if(modelPath != None):
        if(not isinstance(modelPath, list)):
            modelName = Const.ParseModelName(modelPath)
            if(consts.Name() != Const.RemoveUID(modelName)):
                modelName = consts.Name(consts.UID())        
        logSaveDir = saveDir + f"/{modelName}/"

        if(isinstance(modelPath, list)):
            model = FuseModels(modelPath, consts)
        else:
            model, modified = LoadModel(modelPath, consts)
            if(not modified):
                first_epoch = ParseEpoch(modelPath) +1
    else:
        if(consts.Fusion):
            model = FuseModels(None, consts)
        else:
            model = CreateModel(consts.classCount, 1 if consts.noFeature else consts.featureComponents, noColor=consts.noFeature)
    
    if(modelName is None or modelName == ""):
        modelName = consts.Name(consts.UID())
        logSaveDir = saveDir + f"/{modelName}/"

    PrintToLog("Train {} on {} files. Test on {} files".format(modelName, len(trainFiles), len(testFiles)))
    PrintToLog("Validate on :" + str(testFiles))

    trainingSteps = int((1000*16)/consts.batchSize) if not Const.IsWindowsMachine() else int(10)
    PrintToLog("Batch size: {}, trainingSteps: {}".format(consts.batchSize, trainingSteps))

    logsPath = os.path.join(consts.logsPath, Const.RemoveUID(modelName))
    os.makedirs(logsPath, exist_ok=True)
    callbacks_list = []    
    callbacks_list.append(ModelSaveCallback(logSaveDir, trainingSteps, "curb", modelNamePrefix = modelName, sendNotifications=sendNotifications))
    # callbacks_list.append(IOUPerClass(logsPath, consts.classNames[1:], first_epoch+1))
    # callbacks_list.append(tf.keras.callbacks.TensorBoard(log_dir=logsPath, update_freq="batch", histogram_freq=0, profile_batch = 0)) # tensorboard 2.0.2

    seq = TrainSequence(trainFiles, trainingSteps, consts)
    validationSteps = int(((150 if not Const.IsWindowsMachine() else 10) * 16)/consts.batchSize)
    validationData = None if len(testFiles) == 0 else GetValidationData(testFiles, consts, validationSteps)

    if(epochs is None):
        epochs = 20 if consts.Fusion else 100

    model.fit(seq, validation_data = validationData, epochs = epochs, batch_size = consts.batchSize, workers = consts.batchSize, max_queue_size = 300, callbacks=callbacks_list, initial_epoch = first_epoch)

def EvaluateModels(modelsList, testFiles, consts, x = None, y = None):
    if(x is None or y is None):
        validationSteps = int(((150 if not Const.IsWindowsMachine() else 10) * 16)/consts.batchSize)
        x, y = GetValidationData(testFiles, consts, validationSteps, newDataGeneration = False)

    for file in modelsList:
        model, _ = LoadModel(file, consts)
        metrics = model.evaluate(x, y, batch_size = consts.batchSize, workers = consts.batchSize, max_queue_size = 300)
        # print(f"miou: {metrics[2][0][0]*100:.3}")

def SaveModel(saveDir, epoch, model, train_score, val_score=0, modelNamePrefix = ""):
    if(modelNamePrefix != ""):
        modelNamePrefix += "_"
    fileName = saveDir+"/{0}{1}{2}{3}.h5".format(modelNamePrefix, epoch, f"_train({train_score:.3})", f"_val({val_score:.3})" if val_score != 0 else "")
    if(not os.path.isdir(saveDir)):
        os.mkdir(saveDir)      
    if(os.path.exists(fileName)):
        os.remove(fileName)    
    model.save(fileName, include_optimizer=False)

def RotatePointCloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [0, 0, 1],])
    return np.dot(batch_data, rotation_matrix)

def JitterRGB(features):
    features = features.astype(np.uint8)
    assert(np.max(features) > 1)

    img = Image.fromarray(np.expand_dims(features,0), mode="RGB")

    low = 0.4
    high = 1.6
    #1 is baseline
    img = ImageEnhance.Brightness(img).enhance(np.random.uniform(low, high))
    img = ImageEnhance.Color(img).enhance(np.random.uniform(low, high))
    img = ImageEnhance.Contrast(img).enhance(np.random.uniform(low, high))

    img = ImageEnhance.Sharpness(img).enhance(np.random.uniform(low, high))
    if(np.random.uniform(low, high) > 1):
        img = ImageOps.equalize(img)        
    if(np.random.uniform(low, high) > 1):
        img = ImageOps.autocontrast(img)

    new_features = np.array(img).reshape((-1, 3))
    return new_features

def JitterReflectance(features, sigma=40): #input [0; 255]
    assert(features.shape[1] == 1)
    randJitters = np.random.randint(-sigma, sigma, size = features.shape)
    features += randJitters
    features = np.clip(features, 0, 255)
    return features

def JitterPoints(points, sigma=0.01):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """    
    C = 3
    assert(points.shape[1] == C)

    randJitters = np.random.uniform(-sigma, sigma, size = points.shape)
    return points + randJitters

def Mirror(points, axis, min = True):
    if(min):
        axisValue = np.amin(points[:,axis])
    else:
        axisValue = np.amax(points[:,axis])

    distances = np.abs(points[:, axis] - axisValue)
    newpoints = np.array(points, copy=True)

    newpoints[:,axis] = newpoints[:,axis] + distances*(-2 if min else 2)
    return newpoints

def MirrorPoints(points):  
    assert(len(points.shape) == 2 and points.shape[1] == 3)

    mirrorDirection = random.choice(["xMin", "xMax", "yMin", "yMax", ""])

    if(mirrorDirection == "xMin"):
        points = Mirror(points, 0, min = True)
    elif(mirrorDirection == "xMax"):
        points = Mirror(points, 0, min = False)
    elif(mirrorDirection == "yMin"):
        points = Mirror(points, 1, min = True)
    elif(mirrorDirection == "yMax"):
        points = Mirror(points, 1, min = False)
        
    return points

def ScalePoints(points, sigma = 0.02):
    """ Scale up or down random by small percentage
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, scaled batch of point clouds
    """
    assert(points.shape[1]==3)

    scale = np.random.uniform(1-sigma, 1+sigma)
    scale_matrix = np.array([[scale, 0, 0],
                             [0, scale, 0],
                             [0, 0, scale]])
    scaled = np.dot(points, scale_matrix)

    return scaled

class TrainSequence(Sequence):
    def __init__(self, filelist, iteration_number, consts : Const, dataAugmentation = True):
        self.filelist = filelist
        self.ptsList = [np.load(file) for file in self.filelist]
        self.ptsList = sorted(self.ptsList, key=len)
        self.ptsListCount = np.cumsum([len(pts) for pts in self.ptsList])

        self.cts = consts
        self.dataAugmentation = dataAugmentation
        self.iterations = iteration_number

    def __len__(self):
        return int(self.iterations)

    def PickRandomPoint(self, lbl):
        lblIdx = []

        while True:
            randClass = random.randint(0, self.cts.classCount-1)
            lblIdx = np.where(lbl == randClass)[0]

            if(len(lblIdx) >= 2):
                break

        return lblIdx[random.randint(0, len(lblIdx)-1)]        

    def __getitem__(self, _):
        if not self.cts.noFeature:
            ftsList = np.zeros((self.cts.batchSize, self.cts.npoints, self.cts.featureComponents), np.float32)            
        ptsList = np.zeros((self.cts.batchSize, self.cts.npoints, 3), np.float32)
        lbsList = np.zeros((self.cts.batchSize, self.cts.npoints, self.cts.classCount), np.uint8)
        
        for i in range(self.cts.batchSize):
            # load the data
            ptIdx = random.randint(0, self.ptsListCount[-1])
            pts = self.ptsList[np.argmax(self.ptsListCount >= ptIdx)]
            
            # if(self.cts.featureComponents == 1):
            #     keepPts = (pts[:, 4] != 0)
            # else:
            #     keepPts = (pts[:, 6] != 0)
            # pts = pts[keepPts]

            # get the features
            if(self.cts.featureComponents == 1):
                if not self.cts.noFeature:         
                    fts = np.expand_dims(pts[:,3], 1).astype(np.float32)
                lbs = pts[:,4].astype(int)
            else:
                if not self.cts.noFeature:
                    fts = pts[:,3:6].astype(np.float32)
                lbs = pts[:,6].astype(int)

            if(np.min(lbs) == 1):
                lbs -= 1 #class 0 is filtered out
            
            # get the point coordinates
            pts = pts[:, :3]

            # pick a random point
            pt_id = random.randint(0, pts.shape[0]-1)
            pt = pts[pt_id]

            # create the mask
            mask_x = np.logical_and(pts[:,0]<pt[0]+self.cts.blocksize/2, pts[:,0]>pt[0]-self.cts.blocksize/2)
            mask_y = np.logical_and(pts[:,1]<pt[1]+self.cts.blocksize/2, pts[:,1]>pt[1]-self.cts.blocksize/2)
            mask = np.logical_and(mask_x, mask_y)
            temppts = pts[mask]
            templbs = lbs[mask]
            if not self.cts.noFeature:
                tempfts = fts[mask]
            
            # random selection
            choice = np.random.choice(temppts.shape[0], self.cts.npoints, replace=True)
            temppts = temppts[choice]    
            if not self.cts.noFeature:        
                tempfts = tempfts[choice]

            templbs = templbs[choice]
            encodedLbs = np.zeros((len(templbs), self.cts.classCount))
            encodedLbs[np.arange(len(templbs)),templbs] = 1
            templbs = encodedLbs

            # if self.dataAugmentation:
            #     dt = DataTool()
            #     dt.VisualizePointCloudAsync([temppts], [tempfts/255])

            # data augmentation
            if self.dataAugmentation:
                if(self.cts.Mirror):
                    temppts = MirrorPoints(temppts)
                if(self.cts.Rotate):
                    temppts = RotatePointCloud(temppts)
                if(self.cts.Scale):
                    temppts = ScalePoints(temppts, sigma = 0.02)
                if(self.cts.Jitter):
                    temppts = JitterPoints(temppts, sigma = 0.01)

                if(not self.cts.noFeature and self.cts.FtrAugment):
                    if(self.cts.featureComponents == 3):
                        tempfts = JitterRGB(tempfts)
                    elif(self.cts.featureComponents == 1):
                        tempfts = JitterReflectance(tempfts)
                                
            if(not self.cts.noFeature):
                tempfts = tempfts.astype(np.float32)
                tempfts = tempfts/255 # - 0.5
            
            # if self.dataAugmentation:
            #     # visualize data
            #     dt = DataTool()
            #     dt.VisualizePointCloud([temppts], [tempfts], windowName = "Augmented")
            # linePoints = np.where(templbs[:, 1] == 1)[0]
            # DataTool().VisualizePointCloud([np.delete(temppts, linePoints, axis=0), temppts[linePoints]], [[0,0,1], [1,0,0]], windowName="Sampled")

            if not self.cts.noFeature:
                ftsList[i] = np.expand_dims(tempfts, 0)
            ptsList[i] = np.expand_dims(temppts, 0)
            lbsList[i] = np.expand_dims(templbs, 0)
        
        if self.cts.noFeature:
            return [ptsList], lbsList
        else: # works for RGB and fusion models
            return [ftsList, ptsList], lbsList

class TestSequence(Sequence):
    def __init__(self, filename, consts, splitDataSetToParts = -1, windowsMachineCap = True, test = False):
        self.filename = filename
        self.batchSize = consts.batchSize
        self.npoints = consts.npoints
        self.nocolor = consts.noFeature
        self.bs = consts.blocksize
        self.featureComponents = consts.featureComponents
        self.fusion = consts.Fusion
        self.test = test

        if(self.test):
            self.classCount = consts.classCount
            self.lbl = []

        if(self.filename.endswith(".ply")):
            from plyfile import PlyData
            plydata = PlyData.read(self.filename)
            x = plydata["vertex"].data["x"].astype(np.float32)
            y = plydata["vertex"].data["y"].astype(np.float32)
            z = plydata["vertex"].data["z"].astype(np.float32)
            fts = plydata["vertex"].data["reflectance"].astype(np.float32)
            self.xyzrgb = np.concatenate((np.expand_dims(x,1), np.expand_dims(y,1), np.expand_dims(z,1), np.expand_dims(fts, 1)), axis=1)
        elif(self.filename.endswith(".npy")):
            xyzftsl = np.load(self.filename)
            if(xyzftsl.shape[1] == 5):
                self.xyzrgb = xyzftsl[:, :4]
                if(self.test):
                    self.lbl = xyzftsl[:, 4] - 1
            else: #if(xyzftsl.shape[1] == 7):
                self.xyzrgb = xyzftsl[:, :6]
                if(self.test):
                    self.lbl = xyzftsl[:, 6] - 1
        elif(self.filename.endswith(".las")):
            from dataTool import ReadXYZRGB            
            xyz, rgb = ReadXYZRGB(self.filename)
            self.xyzrgb = np.concatenate((xyz, rgb), 1)

        print("Test_step:", consts.test_step)
        step = consts.test_step
        discretized = ((self.xyzrgb[:,:2]).astype(float)/step).astype(int)
        self.allpts = np.unique(discretized, axis=0)
        self.allpts = self.allpts.astype(np.float)*step

        if(consts.IsWindowsMachine() and windowsMachineCap):
            self.allpts = self.allpts[:115] #small sample for testing

        self.splitDataSetToParts = splitDataSetToParts
        if(self.splitDataSetToParts != -1):
            self.ptIndex = 0
        else:
            self.pts = self.allpts
            self.idxList = np.zeros((len(self.pts), self.npoints), np.int64)

        self.sparseCubes = 0
        self.sparseCubesPtCount = 0

    def LenParts(self):
        if(self.splitDataSetToParts != -1):
            return math.ceil(len(self.allpts)/self.splitDataSetToParts)
        else:
            return 1

    def NextPart(self):
        if(self.splitDataSetToParts <= 0):
            return False
        if(self.ptIndex >= len(self.allpts)):
            return False

        self.nextIndex = np.min([self.ptIndex+self.splitDataSetToParts, len(self.allpts)])
        self.pts = self.allpts[self.ptIndex : self.nextIndex]
        self.ptIndex = self.nextIndex

        self.idxList = np.zeros((len(self.pts), self.npoints), np.int64)
        return True

    def __len__(self):
        return math.ceil(len(self.pts)/self.batchSize)

    def compute_mask(self, pt, bs):
        # build the mask
        mask_x = np.logical_and(self.xyzrgb[:,0]<pt[0]+bs/2, self.xyzrgb[:,0]>pt[0]-bs/2)
        mask_y = np.logical_and(self.xyzrgb[:,1]<pt[1]+bs/2, self.xyzrgb[:,1]>pt[1]-bs/2)
        mask = np.logical_and(mask_x, mask_y)
        return mask    

    def __getitem__(self, index):
        size = min(self.batchSize, len(self.pts) - (index * self.batchSize))

        if not self.nocolor:
            ftsList = np.zeros((size, self.npoints, self.featureComponents), np.float32)
        ptsList = np.zeros((size, self.npoints, 3), np.float32)
        if(self.test):
            lblList = np.zeros((size, self.npoints, self.classCount), np.float32)
        
        for i in range(size):
            # get the data            
            mask = self.compute_mask(self.pts[index*self.batchSize + i], self.bs)
            pts = self.xyzrgb[mask]

            if(self.test):
                lbl = self.lbl[mask]

            if(len(pts) < self.npoints):
                self.sparseCubes += 1
                self.sparseCubesPtCount += len(pts)

            # choose right number of points
            choice = np.random.choice(pts.shape[0], self.npoints, replace=True)
            pts = pts[choice]
            if(self.test):
                lbl = lbl[choice]

            # labels will contain indices in the original point cloud
            idx = np.where(mask)[0][choice]
            self.idxList[index*self.batchSize + i] = np.expand_dims(idx, 0)

            # separate between features and points
            if not self.nocolor:
                if(self.featureComponents == 1):
                    fts = np.expand_dims(pts[:,3], 1)
                else:
                    fts = pts[:,3:6]
                fts = fts/255 #- 0.5

            pts = pts[:, :3].copy()

            if not self.nocolor:
                ftsList[i] = np.expand_dims(fts, 0)
            ptsList[i] = np.expand_dims(pts, 0)
            if self.test:
                lblList[i, np.arange(len(lbl)), lbl.astype(int)] = 1

        add_lbl = []
        if self.test:
            add_lbl = [lblList]

        if self.nocolor:
            return [ptsList] + add_lbl
        else: #works for RGB
            return [ftsList, ptsList] + add_lbl

def GenerateData(modelPath, testFiles, consts, outputFolder, NameIncludeModelInfo = False):
    model, _ = LoadModel(modelPath, consts)

    if(not NameIncludeModelInfo):
        outputFolder = os.path.join(outputFolder, Paths.FileName(modelPath))
    os.makedirs(outputFolder, exist_ok=True)

    for file in testFiles:
        t = time()

        baseName = Paths.FileName(file)
        if(NameIncludeModelInfo):
            baseName = baseName + "_" + Paths.FileName(modelPath)
        baseName += ".txt"

        newFile = os.path.join(outputFolder, baseName)
        if(os.path.exists(newFile)):
            print("All ready exists: ",newFile)
            continue
        else:
            open(newFile, "a").close()

        print("Generating: ", newFile)
        GenerateFile(model, file, consts, newFile)
        print("Done in {:02d}:{:02d} min.".format(int((time() - t)/60), int((time() - t)%60)))

def GenerateLargeData(modelPath, voxelFiles, consts, outputFolder, orgFiles = None, replace = False, Upscale = True, NameIncludeModelInfo = False):
    from time import time

    model, _ = LoadModel(modelPath, consts)

    if(not NameIncludeModelInfo):
        outputFolder = outputFolder + Paths.FileName(modelPath)
    if not Upscale:
        outputFolder = outputFolder+"/vox_lbl/"
    os.makedirs(outputFolder, exist_ok=True)

    if isinstance(voxelFiles, str):
        voxelFiles = Paths.GetFiles(voxelFiles)
    
    if isinstance(orgFiles, str):
        orgFiles = Paths.GetFiles(orgFiles)
    
    for voxelFile in voxelFiles:
        baseName = Paths.FileName(voxelFile).replace("_voxels", "")

        if not (orgFiles is None):
            orgFile =  [f for f in orgFiles if Paths.FileName(f).startswith(baseName)]
            if(len(orgFile) != 1):
                print("Skip: ", voxelFile)
                continue
            orgFile = orgFile[0]
        else:
            orgFile = None

        t = time()

        if(NameIncludeModelInfo):
            baseName = baseName + "_" + Paths.FileName(modelPath)
        
        if Upscale:            
            newFile = os.path.join(outputFolder, baseName+".labels")
        else:            
            newFile = os.path.join(outputFolder, baseName+".npy")
        if(not replace and os.path.exists(newFile)):
            print(newFile," already exists.")
            continue
        
        flagFile = newFile+".tmp"
        if(os.path.exists(flagFile)):
            print("Other worker is generating: ", newFile)
            continue
        else:
            open(flagFile, "a").close()

        print("Generating: ", newFile)
        GenerateLargeFile(model, voxelFile, orgFile, consts, newFile, Upscale = Upscale)

        os.remove(flagFile)
        print("{} generated in {:02d}:{:02d} min.".format(baseName, int((time() - t)/60), int((time() - t)%60)))

def GenerateFile(model, file, consts, outputFile, saveScores = True):
    seq = TestSequence(file, consts)
    output = model.predict(seq, workers = consts.batchSize, max_queue_size = 300, verbose = 1)

    # for y in range(len(seq)):
    #     pts = seq.__getitem__(y)
    #     pts = pts[0]
    #     pred = model.predict(pts)

    #     for i in range(len(pred)):
    #         predPtsIdx = np.where(np.argmax(pred[i], axis = 1) == 1)[0]
    #         # truePtsIdx = np.where(np.argmax(lbl[i], axis = 1) == 1)[0]
            
    #         # print(f"True curb points: {len(truePtsIdx)}. Found curb points: {len(predPtsIdx)}")
    #         DataTool().VisualizePointCloud([np.delete(pts[i], predPtsIdx, axis=0), pts[i][predPtsIdx]], [[0,0,1], [1,0,0]])

    idx = seq.idxList
    xyzrgb = seq.xyzrgb[:,:3]
    scores = np.zeros((xyzrgb.shape[0], consts.classCount))

    for i in range(len(output)):
        scores[idx[i]] += output[i]    

    mask = np.logical_not(scores.sum(1)==0)
    scores = scores[mask]
    pts_src = xyzrgb[mask]

    # create the scores for all points
    indexes = nearest_correspondance(pts_src.astype(np.float32), xyzrgb.astype(np.float32), K=1)
    scores = scores[indexes]
    
    if saveScores:
        scoresFile = outputFile.replace(".txt", "_scores.npy")
        np.save(scoresFile, scores)
        print(f"Scores saved to: {scoresFile}")
        
    scores = scores.argmax(1) + 1 #because all class are shifted to avoid 0 - unclassified
    
    print(f"class 0: {len(np.where(scores == 0)[0])}, class 1: {len(np.where(scores == 1)[0])}")

    import pandas as pd
    print("Save labels: ", scores.shape)
    pd.DataFrame(scores, dtype=np.uint8).to_csv(outputFile, sep='\t', header=None, index=None)

def SaveLabelsPnts(labels, outputFile):
    import pandas as pd    
    print("Saving pts lbs...")
    if(len(labels.shape) == 1):
        pd.DataFrame(labels, dtype=np.uint8).to_csv(outputFile, sep='\t', header=None, index=None)
    else:
        np.save(outputFile, labels)
    print("Pts lbs {} saved!".format(labels.shape))

def UpscaleToOriginal(originalPoints, pts_src, lbl, outputFile = None):
    from tqdm import tqdm
    # create the scores for all points
    step = 10000000 #1000000
    fullLbl = np.zeros((0,), np.int8)
    print("KDTree magic. Source pts: {}. Queary pts: {}".format(len(pts_src), len(originalPoints)))
    for i in tqdm(range(0, math.ceil(len(originalPoints)/step))):
        a = i*step
        b = a + np.min([len(originalPoints)-a, step])
        indexes = nearest_correspondance(pts_src, originalPoints[a:b], K=1)
        fullLbl = np.concatenate([fullLbl, lbl[indexes]], 0)

    if(not (outputFile is None)):
        SaveLabelsPnts(fullLbl, outputFile)
    else:
        return fullLbl

def GenerateLargeFile(model, voxelFile, originalFile, consts, outputFile, Upscale = True, saveScores = True):
    from dataTool import ReadXYZ
    from tqdm import tqdm

    seq = TestSequence(voxelFile, consts, splitDataSetToParts=16000)
    print("All pts: ", len(seq.allpts))

    xyzrgb = seq.xyzrgb[:,:3]
    scores = np.zeros((xyzrgb.shape[0], consts.classCount))

    for _ in tqdm(range(seq.LenParts())):
        seq.NextPart()
        output = model.predict(seq, workers = consts.batchSize, max_queue_size = 300, verbose = 1)

        idx = seq.idxList
        for i in range(len(output)):
            scores[idx[i]] += output[i]

    mask = np.logical_not(scores.sum(1)==0)
    scores = scores[mask]
    pts_src = xyzrgb[mask].astype(np.float32)

    if saveScores:
        scoresFile = os.path.splitext(outputFile)[0]+"_scores.npy"
        np.save(scoresFile, scores)
        print(f"Scores saved to: {scoresFile}")

    lbl = scores.argmax(1)
    
    if(Upscale and not (originalFile is None)):
        print("Load original file: ", originalFile)
        originalPoints = ReadXYZ(originalFile).astype(np.float32)
        assert(originalPoints.shape[1] == 3)
        UpscaleToOriginal(originalPoints, pts_src, lbl, outputFile)
    else:        
        SaveLabelsPnts(np.concatenate([pts_src, np.expand_dims(lbl, 1)], axis=1), outputFile)

def UpscaleFilesAsync(modelPath, voxelFolder, orgFolder, savePath):
    import time
    # notifyDevice = Notify()

    savePath = savePath + Paths.FileName(modelPath)

    print(f"Searching in folder: {savePath+'/vox_lbl/'}")

    while True:
        found = False

        fileNames = Semantic3D.fileNames
        for file in Paths.GetFiles(savePath, onlyNames=True, withoutExtension=True, findExtesions=('.labels')):
            if(file in fileNames or fileNames.values()):
                fileNames = {key:val for key, val in fileNames.items() if val != file and key != file}
                    
        if(len(fileNames) == 0):        
            print("Done upscaling files")
            # notifyDevice.send("Done upscaling files")
            return

        for file in Paths.GetFiles(savePath+"/vox_lbl/", onlyNames=True, withoutExtension=True, findExtesions=('.npy')):                
            ptslbs = os.path.join(savePath+"/vox_lbl/", file+".npy")
            # originalFile = os.path.join(orgFolder, file+".npy")
            originalFile = os.path.join(orgFolder, file+".hdf5")
            outputFile = os.path.join(savePath, file+".labels")

            if(not os.path.exists(outputFile)):
                found = True
                open(outputFile, "a").close()
                UpscaleFile(ptslbs, originalFile, outputFile)
        
        if not found:
            time.sleep(10) #sleep for 10 second and scan for job again

def UpscaleFile(ptslbsFile, originalFile, outputFile):
    from dataTool import ReadLabels, ReadXYZ

    print("Upscaling: {}".format(ptslbsFile))
    scores = ReadLabels(ptslbsFile, readFormat = ".npy")
    scores = np.squeeze(scores, 1)
    pts_src = ReadXYZ(ptslbsFile, readFormat = ".npy")
    originalPoints = ReadXYZ(originalFile)

    UpscaleToOriginal(originalPoints, pts_src, scores, outputFile)

def nearest_correspondance(pts_src, pts_dest, K=1):
    # print("KDTree magic. Source pts: {}. Queary pts: {}".format(len(pts_src), len(pts_dest)))
    # t = time()
    kdt = KDTree(pts_src, leaf_size=20)
    _, indexes = kdt.query(pts_dest, k = K)
    # print("Done in {}:{} min.".format(int((time() - t)/60), int((time() - t)%60)))    
    return np.squeeze(indexes, 1)

def TestTestSequence(path, consts):    
    seq = TestSequence(path, consts)

    allPts = np.zeros((len(seq.xyzrgb), 3))

    for i in range(len(seq)):
        inpt = seq[i]

        ftsList = inpt[0]
        ptsList = inpt[1]

        for j in range(len(ptsList)):
            allPts[seq.idxList[i*consts.batchSize + j]] = ptsList[j]
    
    emptyPts = np.logical_not(allPts.sum(1) != 0)

    print("sparseCubes: ",seq.sparseCubes)
    print("mean sparseCubes pt count: ", seq.sparseCubesPtCount/seq.sparseCubes)
    print("Not picked points: {} => {:.2f}%".format(len(emptyPts), len(emptyPts)/len(allPts)))

    nonEmptyPts = np.logical_not(emptyPts)

    a = seq.xyzrgb[emptyPts]
    b = seq.xyzrgb[nonEmptyPts]

    dt = DataTool()
    dt.VisualizePointCloud([a, b], [[1,0,0], None])

if(os.path.exists("C:/Program Files")):
    import open3d as o3d
    import time
    from dataTool import LoadRenderOptions, SaveRenderOptions, GetPointsIndexInBoundingBox, GetPointsInBoundingBox

class BoxesIterator:
    def __init__(self, boxes, points, colors, labels):
        # self.pc = o3d.geometry.PointCloud()
        # self.pc.points = o3d.utility.Vector3dVector(points)
        self.src_points = points
        self.src_colors = colors if np.max(colors) <= 1 else colors/255
        self.src_labels = labels
        self.dst_points = np.zeros((0, 3), dtype = np.float)
        self.dst_colors = np.zeros((0, 3), dtype = np.float)
        self.boxes = boxes
        self.i = 0
        # self.kdt = KDTree(points, leaf_size=20)    

        self.trajectory = None
        # if(os.path.exists("./data/camera_trajectory.json")):
        #     self.trajectory = o3d.io.read_pinhole_camera_trajectory("./data/camera_trajectory.json").parameters
        #     self.trajectory_i = 0
        #     self.trajectory_time = time.time()

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

        #NPM3D
        self.colors = []
        if(np.max(self.src_labels) == 9):
            self.colors = [grey, red, blue, teal, mint, brown, pink, black, green]
        #Semantic3D
        elif(np.max(self.src_labels) == 8):
            self.colors = [grey, verygreen, green, mint, red, blue, brown, black]
        
        self.pc = o3d.geometry.PointCloud()        
        self.pc.points = o3d.utility.Vector3dVector(self.src_points)

        self.box = o3d.geometry.LineSet()
        lines = np.array([[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],[0, 4], [1, 5], [2, 6], [3, 7]])
        self.box.lines = o3d.utility.Vector2iVector(lines)
        self.box.colors = o3d.utility.Vector3dVector(np.array([[1,0,0] for _ in range(len(lines))]))

        self.initSet = False

    def ColorPtsByClass(self, pts, lbl):
        pts_colors = np.zeros((len(pts), 3), np.float)

        for i in range(0, len(self.colors)):
            indexes = np.where(lbl == i+1)[0]
            pts_colors[indexes] = self.colors[i]

        return pts_colors
    
    def BoxPts(self, bBox):
        box =  [[bBox[0], bBox[2], bBox[4]], 
                [bBox[1], bBox[2], bBox[4]], 
                [bBox[0], bBox[3], bBox[4]], 
                [bBox[1], bBox[3], bBox[4]],
                [bBox[0], bBox[2], bBox[5]], 
                [bBox[1], bBox[2], bBox[5]], 
                [bBox[0], bBox[3], bBox[5]], 
                [bBox[1], bBox[3], bBox[5]]]
        return np.array(box)

    def AnimationFunction(self, vis):
        # time.sleep(0.2)
        if(self.i < len(self.boxes)):        
            pts = self.src_points[:, :2]
            mask_x = np.logical_and(self.boxes[self.i][0]<pts[:,0], pts[:,0]<self.boxes[self.i][1])
            mask_y = np.logical_and(self.boxes[self.i][2]<pts[:,1], pts[:,1]<self.boxes[self.i][3])
            ptsIdx = np.where(np.logical_and(mask_x, mask_y))[0]
            randIdx = np.random.choice(ptsIdx, min(8192, len(ptsIdx)), replace=False)
    
            self.dst_points = np.concatenate((self.dst_points, self.src_points[randIdx]), axis = 0)
            self.dst_colors = np.concatenate((self.dst_colors, self.ColorPtsByClass(self.src_points[randIdx], self.src_labels[randIdx])), axis = 0)

            self.src_points = np.delete(self.src_points, randIdx, axis = 0)
            self.src_labels = np.delete(self.src_labels, randIdx, axis = 0)
            self.src_colors = np.delete(self.src_colors, randIdx, axis = 0)
            
            self.pc.points = o3d.utility.Vector3dVector(np.concatenate((self.src_points, self.dst_points), 0))
            self.pc.colors = o3d.utility.Vector3dVector(np.concatenate((self.src_colors, self.dst_colors), 0))

            self.box.points = o3d.utility.Vector3dVector(self.BoxPts(self.boxes[self.i]))

            vis.clear_geometries()
            vis.add_geometry(self.pc, False)
            vis.add_geometry(self.box, False)
                    
            self.i += 1 
            # print(f"{self.i}/{len(self.boxes)}", end="\r")
        else:
            print("Iteration over.")

        if(not os.path.exists("./data/camera_trajectory.json")):
            self.trajectory = None

        if(self.trajectory is None):
            # vis = LoadRenderOptions(vis, returnVis=True)
            if(os.path.exists("./data/camera_trajectory.json")):
                self.trajectory = o3d.io.read_pinhole_camera_trajectory("./data/camera_trajectory.json").parameters
                self.trajectory_i = 0
                self.trajectory_time = time.time()                        
        else:
            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(self.trajectory[self.trajectory_i])
            if(self.trajectory_i < len(self.trajectory)-1): #and time.time() - self.trajectory_time > 1
                print(f"Trajectory: {self.trajectory_i}/{len(self.trajectory)}", end="\r")
                self.trajectory_i += 1
                self.trajectory_time = time.time()

        return False

def ShowSequenceBoxes(ptsFile, lblFile, consts):
    from dataTool import DataTool

    consts.test_step = 4
    seq = TestSequence(ptsFile, consts, windowsMachineCap=False)

    minZ = np.min(seq.xyzrgb[:,2])
    maxZ = np.max(seq.xyzrgb[:,2])

    boxes = []
    for pt in seq.pts:
        minX = pt[0] - consts.blocksize/2
        maxX = pt[0] + consts.blocksize/2
        
        minY = pt[1] - consts.blocksize/2
        maxY = pt[1] + consts.blocksize/2

        boxes.append([minX, maxX, minY, maxY, minZ, maxZ])

    dt = DataTool()
    # dt.VisualizePointCloud([seq.xyzrgb[:,:3]], [seq.xyzrgb[:,3:6]], bBoxes = boxes)
    boxesitr = BoxesIterator(boxes, seq.xyzrgb[:,:3], seq.xyzrgb[:,3:], np.squeeze(ReadLabels(lblFile),1))
    dt.VisualizePointCloud([seq.xyzrgb[:,:3]], animationFunction=boxesitr.AnimationFunction)
    # dt.VisualizePointCloud([seq.xyzrgb[:,:3]])

def RunExperiments():
    from dataTool import VisualizePointCloudClassesAsync, VisualizePointCloudClasses, ReadLabels, DataTool, ReadXYZ
    # testCloud = "G:/PointCloud DataSets/NPM3D/test_10_classes/ajaccio_2.ply"
    # testCloud = consts.Paths.processedTrain+"/Lille1_1_0.npy"
    # VisualizePointCloudClassesAsync(testCloud, downSample=False, windowName="Keras")
    # VisualizePointCloudClassesAsync(testCloud, "G:/PointCloud DataSets/NPM3D/generatedResults/ajaccio_2.txt", downSample=False, windowName="Keras")
    # VisualizePointCloudClassesAsync(testCloud, "G:/PointCloud DataSets/NPM3D/torch_generated_data/results88.2%/ajaccio_2.txt", downSample=False, windowName="Torch")

    # TestTestSequence(consts.Paths.processedTrain+"/Lille1_1_0.npy", consts)
    # ShowSequenceBoxes(consts.Paths.processedTrain+"/Lille1_1_0.npy", consts)

    # # pts = ReadXYZ(consts.Paths.processedTrain+"/Lille2_0.npy")
    # true = ReadLabels(consts.Paths.processedTrain+"/Lille2_0.npy")

    # # pts = ReadXYZ(consts.Paths.rawTrain+"/untermaederbrunnen_station3_xyz_intensity_rgb.hdf5")
    # # true = ReadLabels(consts.Paths.rawTrain+"/untermaederbrunnen_station3_xyz_intensity_rgb.hdf5")

    # # pred_file = "G:/PointCloud DataSets/NPM3D/torch_generated_data/results88.2%/Lille2_0.txt"
    # pred_file = consts.Paths.generatedTest+"/"+Paths.FileName(modelPath)+"/Lille2_0.txt"
    # # pred_file = consts.Paths.generatedTest+"/"+Paths.FileName(modelPath)+"/untermaederbrunnen_station3_xyz_intensity_rgb.labels"
    # pred = ReadLabels(pred_file)
    
    # VisualizePointCloudClasses(consts.Paths.processedTrain+"/Lille2_0.npy",
    #                             pred_file,
    #                             downSample=False, windowName="Red error",
    #                             errorPoints = ((true != pred) == (true != 0)),
    #                             delPoints = (true == 0))

    # error = np.where(true == 0)[0]
    # true = np.delete(true, error, 0)
    # pred = np.delete(pred, error, 0)

    # from sklearn.metrics import confusion_matrix
    # import metrics
    # cm = confusion_matrix(true, pred, labels=list(range(consts.classCount)))
    # iou =  metrics.stats_iou_per_class(cm)
    # print("Mean iou:", iou[0])
    # print("iou per class:", iou[1])

    from dataTool import ReadXYZ, ReadLabels
    from sklearn.metrics import confusion_matrix
    from metrics import stats_accuracy_per_class, stats_iou_per_class

    src_pts = ReadXYZ(r"G:\PointCloud DataSets\semantic3d\rawTrain\bildstein_station3_xyz_intensity_rgb.hdf5")
    src_lbl = ReadLabels(r"G:\PointCloud DataSets\semantic3d\rawTrain\bildstein_station3_xyz_intensity_rgb.hdf5")
    src_lbl = np.squeeze(src_lbl, 1)

    delIndices = np.where(src_lbl == 0)
    src_pts = np.delete(src_pts, delIndices, axis=0)
    src_lbl = np.delete(src_lbl, delIndices, axis=0)

    voxel_pts = ReadXYZ(r"G:\PointCloud DataSets\semantic3d\processedTrain(0.15m)\bildstein_station3_xyz_intensity_rgb_voxels.npy")
    voxel_lbl = ReadLabels(r"G:\PointCloud DataSets\semantic3d\processedTrain(0.15m)\bildstein_station3_xyz_intensity_rgb_voxels.npy")
    voxel_lbl = np.squeeze(voxel_lbl, 1)

    upscaled_lbl = UpscaleToOriginal(src_pts, voxel_pts, voxel_lbl)

    cm = confusion_matrix(src_lbl, upscaled_lbl)
    avg_acc, avg_class = stats_accuracy_per_class(cm)
    avg_iou, avg_iou_class = stats_iou_per_class(cm)

def RenameSemantic3DFiles(folder):
    if(len(Paths.GetFiles(folder, findExtesions = ".labels")) == 0):
        print("No files found.")
        return

    for file in Paths.GetFiles(folder, findExtesions = ".labels"):
        if(Paths.FileName(file).endswith("(1)")):
            os.remove(file)
        else:
            name = Paths.FileName(file)
            newFileName = file.replace(name, Semantic3D.fileNames[name])
            os.rename(file, newFileName)

            if(os.path.getsize(newFileName) == 0):
                print(f"{newFileName} if 0 bytes size")
    
    if(len(Paths.GetFiles(folder, findExtesions = ".labels")) != 15):
        print("Wrong number of files.")
    else:
        print("Done renaming: ", folder)

if __name__ == "__main__":
    from NearestNeighbors import NearestNeighborsLayer, SampleNearestNeighborsLayer
    from KDTree import KDTreeLayer, KDTreeSampleLayer
    modelPath = None

    # consts = NPM3D()
    # consts = Semantic3D()
    consts = Curbs()

    consts.noFeature = True
    # consts.Fusion = True
    # consts.Scale = True
    consts.Rotate = True
    # consts.Mirror = True
    # consts.Jitter = True
    # consts.FtrAugment = True

    testFiles = consts.TestFiles()
    trainFiles = consts.TrainFiles()

    modelPath = "Sem3D(vox)(fusion)(FullAugment)_3_train(86.2)_val(79.5).h5"
    # modelPath = "Curbs(7&1)(noFeature)(Rotate)_21bdbe6aa82d4e259526ab46577e795a_25_train(75.1)_val(60.7).h5"
    # modelPath = ["Sem3D(vox)(RGB)(FullAugment)_55_train(85.7)_val(79.9)", "Sem3D(NOCOL)_50_train(87.4)_val(69.1)"]
    # modelPath = ["NPM3D(80&5)(RGB)(NoScale)_28_train(88.3)_val(73.2).h5", "NPM3D(80&5)(NOCOL)(FullAugment)_28_train(87.3)_val(71.5).h5"]
    # modelPath = LatestModel("Sem3D(14&1)(noFeature)(Scale)(Rotate)(Mirror)(Jitter)")
    # modelPath = LatestModel(consts.Name())    

    if(isinstance(modelPath,list)):
        consts.Fusion = True

    if(not consts.Fusion and not Const.IsWindowsMachine()):
        tf.config.optimizer.set_jit(True) #Gives more than 10% boost!!!
        print("XLA enabled.")

    # modelPath = ["Sem3D(14&1)(noFeature)(Scale)(Rotate)(Mirror)(Jitter)_9bbee708a7814063af9d85070452abd8_59_train(85.2)_val(72.8)", 
    #             "Sem3D(14&1)(noFeature)(Rotate)(Mirror)(Jitter)_ff2eb229084247d9a1c63caa519e9890_58_train(84.9)_val(75.5)",
    #             "Sem3D(14&1)(noFeature)_dffc17f77e924894bbdbdad818ab6994_40_train(85.1)_val(68.8)"]
    # EvaluateModels([modelPath], testFiles, consts)

    TrainModel(trainFiles, testFiles, consts, modelPath = modelPath)# , epochs = 8) #continue train
    # TrainModel(trainFiles, testFiles, consts) #new model

    # modelPath = HighestValMIOUModel("NPM3D(80&5)(fusion)(FullAugment)")

    #NPM3D
    # GenerateData(modelPath, Paths.GetFiles(consts.Paths.rawTest), consts, consts.Paths.generatedTest)
    
    #Semantic3D    
    # GenerateLargeData(modelPath, Paths.Semantic3D.processedTest, Paths.Semantic3D.rawTest, consts, consts.Paths.generatedTest, Upscale=False)
    # UpscaleFilesAsync(modelPath, Paths.Semantic3D.processedTest, Paths.Semantic3D.rawTest, Paths.Semantic3D.generatedTest)
    # RenameSemantic3DFiles(Paths.Semantic3D.generatedTest + Paths.FileName(modelPath))

    #Curbs
    EvaluateModels([modelPath], testFiles, consts)
    # GenerateData(modelPath, testFiles, consts, consts.Paths.pointCloudPath+"/generated/")
    GenerateLargeData(modelPath, testFiles, consts, consts.Paths.pointCloudPath+"/generated/")