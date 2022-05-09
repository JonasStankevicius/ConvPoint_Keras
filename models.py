import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Dense, Dropout, InputLayer
from KDTree import KDTreeLayer, KDTreeSampleLayer, knn_farthest_dll_exists
import json

from configs import *
from imports import *
from metrics import MIOU, IOU

class WeightsMul(tf.keras.layers.Layer):
    def __init__(self, shape, lowBound, highBound, setName = None, **kwargs):
        super(WeightsMul, self).__init__(**kwargs)
        self.shape = shape
        self.lowBound = lowBound
        self.highBound = highBound
        self.setName = "" if (setName is None) else setName

    def build(self, input_shape):
        init = tf.random_uniform_initializer(self.lowBound, self.highBound)
        self.vars = self.add_weight(shape=(self.shape), 
                                    initializer = init, 
                                    trainable = True, dtype=tf.float32, name=self.setName+"WeightsMulWeights")

    def call(self, inputs):        
        return tf.matmul(inputs, self.vars)
    
    def get_config(self):
        config = super(WeightsMul, self).get_config()
        config.update({'shape': self.shape, 'lowBound': self.lowBound, 'highBound': self.highBound})
        return config

class AddBias(tf.keras.layers.Layer):
    def __init__(self, shape, setName = "", **kwargs):
        super(AddBias, self).__init__(**kwargs)
        self.shape = shape
        self.setName = "" if (setName is None) else setName

    def build(self, input_shape):
        self.bias = self.add_weight(shape=self.shape, initializer=tf.keras.initializers.Zeros(), trainable = True, dtype=tf.float32, name=self.setName+"AddBiasWeights")

    def call(self, inputs):        
        return inputs + self.bias
    
    def get_config(self):
        config = super(AddBias, self).get_config()
        config.update({'shape': self.shape})
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
    def __init__(self, dim, n_centers, setName = "", **kwargs):
        super(SubstractCenters, self).__init__(**kwargs)
        self.dim = dim
        self.n_centers = n_centers
        self.setName = "" if (setName is None) else setName
    
    def build(self, input_shape):
        center_data = np.zeros((self.dim, self.n_centers))
        for i in range(self.n_centers):
            coord = np.random.rand(self.dim)*2 - 1
            while (coord**2).sum() > 1:
                coord = np.random.rand(self.dim)*2 - 1
            center_data[:,i] = coord

        self.centers = self.add_weight(shape = (center_data.shape), 
                                        initializer = tf.constant_initializer(center_data), 
                                        trainable = True, dtype=tf.float32, name=self.setName+"SubstractCentersWeights")

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
    
def PtConv(fts, points, K, next_pts, out_features, n_centers = 16, use_bias = True, batchNorm = True, outFtsName = None, outPtsName = None, setName = None, farthest_sampling = False):
    next_pts_ = None
    in_features = fts.shape[-1] if not fts is None else 1

    if isinstance(next_pts, int) and points.get_shape()[1] != next_pts:
        # convolution with reduction    
        farthest_sampling = farthest_sampling and knn_farthest_dll_exists
        indices, next_pts_ = KDTreeSampleLayer(K, next_pts, farthest_sampling = farthest_sampling)(points)
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
    
    if fts is None:
        # get the features and point cooridnates associated with the indices
        pts = tf.gather_nd(points, tf.expand_dims(indices, -1), batch_dims=1)
        features = tf.expand_dims(tf.ones_like(pts[:,:,:,0]), 3)
    else:
        composite = tf.gather_nd(tf.concat((points, fts), 2), tf.expand_dims(indices, -1), batch_dims=1)
        pts, features = tf.split(composite, (3, fts.shape[-1]), axis=3)

    #### End of point sampling
    #### Convolution computation    

    # center the neighborhoods
    pts = pts - tf.expand_dims(next_pts,2)

    # normalize to unit ball, or not
    pts = UnitBallNormalize()(pts)

    # compute the distances
    dists = SubstractCenters(3, n_centers, None if setName is None else setName+"SCW1")(tf.expand_dims(pts, 4))

    dShape = dists.shape
    dists = tf.reshape(dists, (-1, dShape[1], dShape[2], dShape[3]*dShape[4]))

    dists = DenseInitialized(2*n_centers, activation="relu", name = None if setName is None else setName+"Dense1")(dists)
    dists = DenseInitialized(n_centers, activation="relu", name = None if setName is None else setName+"Dense2")(dists)
    dists = DenseInitialized(n_centers, activation="relu", name = None if setName is None else setName+"Dense3")(dists)
    
    # compute features    
    fs = features.shape # [batch, points, n_centers, in_features]
    ds = dists.shape

    features = tf.transpose(features,[0, 1, 3, 2])
    features = tf.reshape(features, (-1, features.shape[2], features.shape[3])) #features.shape[0]*features.shape[1]
    dists = tf.reshape(dists, (-1, dists.shape[2], dists.shape[3])) #dists.shape[0]*dists.shape[1]

    features = tf.matmul(features, dists)
    features = tf.reshape(features, (-1, ds[1], features.shape[1]*features.shape[2]))

    bound = sqrt(3.0) * sqrt(2.0 / (in_features + out_features))
    features = WeightsMul([in_features * n_centers, out_features], -bound, bound, name = None if setName is None else setName+"WM1")(features)

    features = features / fs[2]

    if(use_bias):
        features = AddBias(out_features, name = None if setName is None else setName+"BIAS1")(features)

    # normalization and activation
    if(batchNorm):        
        features = BatchNormalization(epsilon = 1e-05, momentum=0.9, name = None if setName is None else setName+"BatchNorm")(features)

    features = tf.nn.relu(features) if (outFtsName is None) else tf.nn.relu(features, name=outFtsName)
    if(not outPtsName is None):
        next_pts = tf.reshape(next_pts, shape=[-1, next_pts.shape[1], next_pts.shape[2]], name=outPtsName)

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

def ModifyModelOutput(model, classCount):
    dropoutLayer = model.layers[len(model.layers)-5] #take output of the drop out layer
    out_labels = dropoutLayer.output

    out_labels = tf.reshape(out_labels, (-1, out_labels.shape[2]), name = "lbl_reshape_1")
    out_labels = DenseInitialized(classCount, name = "lbl_dense")(out_labels)    
    out_labels = tf.reshape(out_labels, (-1, dropoutLayer.input.shape[1], out_labels.shape[1]), name = "lbl_reshape_2")
    out_labels = tf.nn.softmax(out_labels, name = "lbl_softmax")

    return Model(model.inputs, out_labels, name ="model")

def ReadModel(modelPath):
    if(not modelPath.endswith(".h5") and not os.path.isdir(modelPath)):
        modelPath += ".h5"

    if(not os.path.exists(modelPath) and not os.path.isdir(modelPath)):
        if(os.path.exists(os.path.join("." , "data", modelPath))):
            modelPath = os.path.join("." , "data", modelPath)
        elif(os.path.exists(os.path.join("." , "data", Config.ParseModelName(modelPath, False)))):
            file = os.path.basename(modelPath)
            folder = os.path.join("." , "data", Config.ParseModelName(modelPath, False))
            modelPath = os.path.join(folder, file)
        elif(os.path.exists(os.path.join("." , "data", Config.ParseModelName(modelPath)))):
            file = os.path.basename(modelPath)
            folder = os.path.join("." , "data", Config.ParseModelName(modelPath))
            modelPath = os.path.join(folder, file)

        if(not os.path.exists(modelPath)):
            raise FileNotFoundError    

    model = tf.keras.models.load_model(modelPath, compile=False,
        custom_objects={
                        'SubstractCenters': SubstractCenters,
                        'WeightsMul': WeightsMul,
                        'AddBias': AddBias,
                        'GatherNDLayer':GatherNDLayer,
                        'UnitBallNormalize':UnitBallNormalize,
                        'KDTreeSampleLayer':KDTreeSampleLayer,
                        'KDTreeSampleLayerFast':KDTreeSampleLayer,
                        'KDTreeLayer':KDTreeLayer,
                        'KDTreeLayerFast':KDTreeLayer,
                        })

    return model

def LatestModel(path):
    if(Config.ParseModelUID(path) is None):
        folders = [os.path.join("." , "data",folder) for folder in os.listdir(os.path.join("." , "data")) 
                                                        if os.path.isdir(os.path.join("." , "data",folder)) 
                                                        and path == Config.RemoveUID(Config.ParseModelName(folder))
                                                        and len(Paths.GetFiles(os.path.join("." , "data",folder), findExtesions=".h5")) > 0]
        path = max(folders, key=os.path.getctime)
    else:
        path = os.path.join("." , "data", Config.ParseModelName(path))    

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

def ParseEpoch(modelPath):
    filename = os.path.basename(modelPath)
    return int(filename.split("_")[2])

def SaveMetaInfo(path, metaInfo):
    metaInfoFileName = 'meta-info.json'
    filePath = os.path.join(path, metaInfoFileName)

    with open(filePath, 'w') as jsonFile:
        json.dump(metaInfo, jsonFile)

def SaveModel(saveDir, epoch, model, train_score : float, val_score : float, modelNamePrefix = "", saveTFFormat = False, metaInfo = None):
    if(modelNamePrefix != ""):
        modelNamePrefix += "_"
    path = saveDir+"/{0}{1}{2}{3}".format(modelNamePrefix, epoch, f"_train({train_score:.3})", f"_val({val_score:.3})" if val_score != 0 else "")
    fileName = path+".h5"
    if(not os.path.isdir(saveDir)):
        os.mkdir(saveDir)      
    if(os.path.exists(fileName)):
        os.remove(fileName)    
    model.save(fileName, include_optimizer=False)
    
    if(saveTFFormat):
        model.save(path, save_format="tf", include_optimizer=False)
        if metaInfo:
            SaveMetaInfo(path, metaInfo)

def CreateModel(consts : Config, in_fts = None, in_pts = None, returnFeatures = False, applySoftmax = True):
    print("Creating new model...")
    
    if(in_fts is None and in_pts is None):
        in_pts = Input(shape=(consts.npoints, consts.pointComponents), dtype=tf.float32) #points 

        if(consts.noFeature):
            in_fts = None
        else:
            in_fts = Input(shape=(consts.npoints, consts.featureComponents), dtype=tf.float32) #featuress        
    
    if(consts.noFeature):
        in_fts = None

    pl = consts.pl
    ### Down Sample
    x0, _    = PtConv(in_fts, in_pts,   K = 16, next_pts = None,    out_features = pl)
    
    if(consts.npoints > 8192):
        x11, pts11 = PtConv(x0, in_pts, K = 16, next_pts = 8192,    out_features = pl)
    else:
        x11, pts11 = x0, in_pts
       
    x1, pts1 = PtConv(x11, pts11,       K = 16, next_pts = 2048,    out_features = pl)
    x2, pts2 = PtConv(x1, pts1,         K = 16, next_pts = 1024,    out_features = pl)
    x3, pts3 = PtConv(x2, pts2,         K = 16, next_pts = 256,     out_features = pl)
    x4, pts4 = PtConv(x3, pts3,         K = 8,  next_pts = 64,      out_features = pl*2)
    x5, pts5 = PtConv(x4, pts4,         K = 8,  next_pts = 16,      out_features = pl*2)
    x6, pts6 = PtConv(x5, pts5,         K = 4,  next_pts = 8,       out_features = pl*2)

    ## Up Sample
    x5d, _ = PtConv(x6, pts6, K = 4, next_pts = pts5, out_features = pl*2)
    x5d = tf.concat([x5d, x5], axis = 2)

    x4d, _ = PtConv(x5d, pts5, K = 4, next_pts = pts4, out_features = pl*2)
    x4d = tf.concat([x4d, x4], axis = 2)

    x3d, _ = PtConv(x4d, pts4, K = 4, next_pts = pts3, out_features = pl)
    x3d = tf.concat([x3d, x3], axis = 2)

    x2d, _ = PtConv(x3d, pts3, K = 8, next_pts = pts2, out_features = pl)
    x2d = tf.concat([x2d, x2], axis = 2)

    x1d, _ = PtConv(x2d, pts2, K = 8, next_pts = pts1, out_features = pl)
    x1d = tf.concat([x1d, x1], axis = 2)
    
    x0d, _ = PtConv(x1d, pts1, K = 8, next_pts = pts11, out_features = pl)
    x0d = tf.concat([x0d, x11], axis = 2)
    
    if(consts.npoints > 8192):
        x0d, _ = PtConv(x0d, pts11, K = 8, next_pts = in_pts, out_features = pl)
        x0d = tf.concat([x0d, x0], axis = 2)
  
    ### Output layer
    out_labels = Dropout(rate=0.5)(x0d)
    
    out_labels = tf.reshape(out_labels, (-1, out_labels.shape[2]))
    
    out_labels = DenseInitialized(consts.classCount)(out_labels)

    out_labels = tf.reshape(out_labels, (-1, x0d.shape[1], out_labels.shape[1]))

    if(applySoftmax):
        out_labels = tf.nn.softmax(out_labels)

    if(consts.noFeature):
        inputList = [in_pts]
    else:
        inputList = [in_fts, in_pts]

    if(returnFeatures):
        return Model(inputList, [x0d, out_labels], name ="model")
        
    model = Model(inputList, out_labels, name ="model")
    model = CompileModel(model, consts.classCount, consts.classNames, consts.class_weights)
    # model.summary()
    return model

def LargeScaleParseModel(sparseFts, sparsePts, smallScaleBlocks, pl):
    sparsePts = tf.reshape(sparsePts, (-1, smallScaleBlocks*sparsePts.shape[1], sparsePts.shape[-1])) # [Batch*9, pts, 3] -> [Batch, 9*pts, 3]
    sparseFts = tf.reshape(sparseFts, (-1, smallScaleBlocks*sparseFts.shape[1], sparseFts.shape[-1]))
    
    normalization = True
    
    # LARGE SCALE SPARSE
    x0, _    = PtConv(sparseFts, sparsePts,   K = 8, next_pts = None,    out_features = pl*2, batchNorm = normalization, setName="LargeConv0")
    x1, pts1 = PtConv(x0, sparsePts,       K = 8, next_pts = 64,    out_features = pl*2, batchNorm = normalization, setName="LargeConv1")
    x2, pts2 = PtConv(x1, pts1,         K = 8, next_pts = 16,    out_features = pl*2, batchNorm = normalization, setName="LargeConv2")

    ## Up Sample
    x1d, _ = PtConv(x2, pts2, K = 8, next_pts = pts1, out_features = pl*2, batchNorm = normalization, setName="LargeUpConv3")
    x1d = tf.concat([x1d, x1], axis = 2)

    x0d, _ = PtConv(x1d, pts1, K = 8, next_pts = sparsePts, out_features = pl*2, batchNorm = normalization, setName="LargeUpConv4")
    sparseFts = tf.concat([x0d, x0], axis = 2)
    # END LARGE SCALE SPARSE
                
    sparsePts = tf.tile(sparsePts, [smallScaleBlocks, 1, 1]) # [Batch, 576, 3] -> [Batch*9, 576, 3]
    sparseFts = tf.tile(sparseFts, [smallScaleBlocks, 1, 1])
    
    return sparsePts, sparseFts

def CreateTwoScaleModel(consts : Config):
    print("Creating new 2-scale model...")
    
    # ptsLarge - LARGE SCALE sparse
    # ptsSmall - SMALL SCALE dense

    smallScaleBlocks = consts.input_tile_count
    pl = Config.pl #int(Config.pl/2)    
    
    ptsSmall = Input(shape=(smallScaleBlocks, consts.npoints, consts.pointComponents), dtype=tf.float32) 
    ftsSmall = Input(shape=(smallScaleBlocks, consts.npoints, consts.featureComponents), dtype=tf.float32) #features        
    
    inputList = [ftsSmall, ptsSmall]

    ptsSmall = tf.reshape(ptsSmall, (-1, consts.npoints, consts.pointComponents), name = "large_reshape") # [Batch, 9, npoints, 3] -> [Batch*9, npoints, 3]
    ftsSmall = tf.reshape(ftsSmall, (-1, consts.npoints, consts.featureComponents), name = "large_reshape")
    
    ### Down Sample SMALL SCALE
    x0, _    = PtConv(ftsSmall,ptsSmall,K = 16, next_pts = None,    out_features = pl)
    x1, pts1 = PtConv(x0, ptsSmall,     K = 16, next_pts = 2048,    out_features = pl)
    x2, pts2 = PtConv(x1, pts1,         K = 16, next_pts = 1024,    out_features = pl)
    x3, pts3 = PtConv(x2, pts2,         K = 16, next_pts = 256,     out_features = pl)
    x4, pts4 = PtConv(x3, pts3,         K = 8,  next_pts = 64,      out_features = pl*2)
    x5, pts5 = PtConv(x4, pts4,         K = 8,  next_pts = 16,      out_features = pl*2)
    
    with tf.name_scope("large_process_scope") as scope:
        pts6, x6 = LargeScaleParseModel(x5, pts5, smallScaleBlocks, pl) # [Batch*9, 64, 3]
    
    x5d, _ = PtConv(x6, pts6, K = 4, next_pts = pts5, out_features = pl*2)
    x5d = tf.concat([x5d, x5], axis = 2)

    x4d, _ = PtConv(x5d, pts5, K = 4, next_pts = pts4, out_features = pl*2)
    x4d = tf.concat([x4d, x4], axis = 2)

    x3d, _ = PtConv(x4d, pts4, K = 4, next_pts = pts3, out_features = pl)
    x3d = tf.concat([x3d, x3], axis = 2)

    x2d, _ = PtConv(x3d, pts3, K = 8, next_pts = pts2, out_features = pl)
    x2d = tf.concat([x2d, x2], axis = 2)

    x1d, _ = PtConv(x2d, pts2, K = 8, next_pts = pts1, out_features = pl)
    x1d = tf.concat([x1d, x1], axis = 2)

    x0d, _ = PtConv(x1d, pts1, K = 8, next_pts = ptsSmall, out_features = pl)
    x0d = tf.concat([x0d, x0], axis = 2)

    x0d = tf.reshape(x0d, (-1, smallScaleBlocks, consts.npoints, x0d.shape[-1]), name = "large_reshape") # [Batch*9, npoints, 3] -> [Batch, 9, npoints, 3]
        
    out_labels = Dropout(rate=0.5)(x0d)
    out_labels = tf.reshape(out_labels, (-1, out_labels.shape[-1]))
    out_labels = DenseInitialized(consts.classCount)(out_labels)
    out_labels = tf.reshape(out_labels, (-1, x0d.shape[1], x0d.shape[2], out_labels.shape[-1]))
    out_labels = tf.nn.softmax(out_labels)

    model = Model(inputList, out_labels, name ="model")
    model = CompileModel(model, consts.classCount, consts.classNames)
    # model.summary()
    return model

def CreateTwoScaleModelPretrained(modelPath, consts : Config):
    
    # pretrainedModel, _ = LoadModel(modelPath, consts)
    # pretrainedModel.summary(print_fn=lambda x: open('modelsummary.txt', 'a').write(x + '\n'))
    
    newModel = CreateTwoScaleModel(consts)
    # newModel.load_weights(modelPath)
    newModel.load_weights(modelPath, by_name = True, skip_mismatch=True)
    
    # for new_layer, layer in zip(newModel.layers, pretrainedModel.layers):      
    #     new_layer.set_weights(layer.get_weights())

    # model = Model(newModel.inputs, newModel.outputs, name ="model")
    model = CompileModel(newModel, consts.classCount, consts.classNames)
    # model.summary()
    return model

def CreateModelSmartMerge(consts : Config):
    
    smallScaleBlocks = consts.input_tile_count
    pl = Config.pl #int(Config.pl/2)
    
    in_pts = Input(shape=(smallScaleBlocks, consts.npoints, consts.pointComponents), dtype=tf.float32) 
    in_fts = Input(shape=(smallScaleBlocks, consts.npoints, consts.featureComponents), dtype=tf.float32) #features        
    # in_pts = Input(shape=(smallScaleBlocks, consts.npoints, consts.pointComponents), batch_size = consts.BatchSize(), dtype=tf.float32) 
    # in_fts = Input(shape=(smallScaleBlocks, consts.npoints, consts.featureComponents), batch_size = consts.BatchSize(), dtype=tf.float32) #features        
    
    inputList = [in_fts, in_pts]

    in_pts = tf.reshape(in_pts, (-1, consts.npoints, consts.pointComponents)) # [Batch, 9, npoints, 3] -> [Batch*9, npoints, 3]
    in_fts = tf.reshape(in_fts, (-1, consts.npoints, consts.featureComponents))    
    
    x0, _    = PtConv(in_fts, in_pts,   K = 16, next_pts = None,    out_features = pl)
    x1, pts1 = PtConv(x0, in_pts,       K = 16, next_pts = 2048,    out_features = pl)
    x2, pts2 = PtConv(x1, pts1,         K = 16, next_pts = 1024,    out_features = pl)
    x3, pts3 = PtConv(x2, pts2,         K = 16, next_pts = 256,     out_features = pl)
    x4, pts4 = PtConv(x3, pts3,         K = 8,  next_pts = 64,      out_features = pl*2)
    x5, pts5 = PtConv(x4, pts4,         K = 8,  next_pts = 16,      out_features = pl*2)
    x6, pts6 = PtConv(x5, pts5,         K = 4,  next_pts = 8,       out_features = pl*2)

    ## Up Sample
    x5d, _ = PtConv(x6, pts6, K = 4, next_pts = pts5, out_features = pl*2)
    x5d = tf.concat([x5d, x5], axis = 2)

    x4d, _ = PtConv(x5d, pts5, K = 4, next_pts = pts4, out_features = pl*2)
    x4d = tf.concat([x4d, x4], axis = 2)

    x3d, _ = PtConv(x4d, pts4, K = 4, next_pts = pts3, out_features = pl)
    x3d = tf.concat([x3d, x3], axis = 2)

    x2d, _ = PtConv(x3d, pts3, K = 8, next_pts = pts2, out_features = pl)
    x2d = tf.concat([x2d, x2], axis = 2)

    x1d, _ = PtConv(x2d, pts2, K = 8, next_pts = pts1, out_features = pl)
    x1d = tf.concat([x1d, x1], axis = 2)
    
    x1d = tf.reshape(x1d, (-1, smallScaleBlocks*x1d.shape[1], x1d.shape[-1])) # [Batch*9, 2048, 3] -> [Batch, 9*2048, 3]
    pts1 = tf.reshape(pts1, (-1, smallScaleBlocks*pts1.shape[1], pts1.shape[-1])) # [Batch*9, 2048, 3] -> [Batch, 9*2048, 3]
    
    allPts = tf.tile(pts1, [smallScaleBlocks, 1, 1]) # [Batch, 576, 3] -> [Batch*9, 576, 3]
    allFts = tf.tile(x1d, [smallScaleBlocks, 1, 1])

    x0d, _ = PtConv(allFts, allPts, K = 8, next_pts = in_pts, out_features = pl)
    x0d = tf.concat([x0d, x0], axis = 2)
  
    # x0d = tf.reshape(x0d, (-1, smallScaleBlocks, consts.npoints, x0d.shape[-1])) # [Batch*9, npoints, 3] -> [Batch, 9, npoints, 3]        
    ### Output layer    
    out_labels = Dropout(rate=0.5)(x0d)
    out_labels = tf.reshape(out_labels, (-1, out_labels.shape[-1]))
    out_labels = DenseInitialized(consts.classCount)(out_labels)
    out_labels = tf.reshape(out_labels, (-1, smallScaleBlocks, consts.npoints, consts.classCount))
    out_labels = tf.nn.softmax(out_labels)

    model = Model(inputList, out_labels, name ="model")
    model = CompileModel(model, consts.classCount, consts.classNames)
    # model.summary()
    return model

def LoadModel(modelPath, consts):
    model = ReadModel(modelPath)

    modified = False
    if(model.output.shape[2] != consts.classCount):
        print("Model output {} classes changed to {}".format(model.output.shape[2], consts.classCount))
        modified = True
        model = ModifyModelOutput(model, consts.classCount)

    model = CompileModel(model, consts.classCount, consts.classNames, None)
    # model.summary()
    return model, modified

def ReadModelConfig(path):
    Model = ReadModel(path)
    modelConfig = ParseModelConfig(path)
    return Model, modelConfig

def CreateModelCopy(Model, modelConfig, in_pts, in_RGB):
    inputFeatures = 1 if modelConfig.noFeature else modelConfig.featureComponents
    newModel = CreateModel(modelConfig, in_RGB, in_pts, returnFeatures=True, applySoftmax=False)

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
        in_RGB = Input(shape=(consts.npoints, consts.featureComponents), dtype=tf.float32, name = "In_RGB") #features
    in_pts = Input(shape=(consts.npoints, Config.pointComponents), dtype=tf.float32, name = "In_pts") #points

    newModelA = CreateModelCopy(ModelA, modelAConfig, in_pts, in_RGB)
    newModelB = CreateModelCopy(ModelB, modelBConfig, in_pts, in_RGB)

    x = tf.concat((newModelA.output[0], newModelB.output[0]), axis = 2) #fuse features from both models

    x1, _    = PtConv(x, in_pts,   K = 16, next_pts = consts.npoints,    in_features = 2*128,  out_features = 96)
    x2, _    = PtConv(x1, in_pts,   K = 16, next_pts = consts.npoints,    in_features = 96,  out_features = 48)
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

    fusionModel = CompileModel(fusionModel, consts.classCount)
    # fusionModel.summary()
    return fusionModel

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = tf.constant(weights, dtype=tf.float32)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        # calc
        loss = tf.cast(y_true, tf.float32) * tf.math.log(y_pred) * weights
        loss = -tf.reduce_sum(loss, -1)
        loss = tf.math.reduce_mean(loss)
        return loss
    
    return loss

def CompileModel(model, classCount, classNames, classWeights):
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, epsilon = 1e-8),
        loss = tf.keras.losses.CategoricalCrossentropy(),
        # loss = weighted_categorical_crossentropy(classWeights),
        # metrics= [IOU(classCount, 0, name="other"), IOU(classCount, 1, name="curb")] if classCount == 2 else [MIOU(classCount)]
        # metrics= [MIOU(classCount), tf.keras.metrics.CategoricalAccuracy(name='accuracy')]  + [IOU(classCount, i, name=name) for i, name in enumerate(classNames)]
        metrics= [MIOU(classCount), tf.keras.metrics.CategoricalAccuracy(name='accuracy')]
    )
    return model