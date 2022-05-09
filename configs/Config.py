from dataTool import ReadXYZRGBLBL, ProcessFeatures, RotatePointCloud, PickPointsInBlock, DataTool, BoundingBoxFromVoxel
from imports import *
from enum import Enum
import tensorflow as tf
from tqdm import tqdm

class DataPipeline(Enum):
    Sequence = 1
    tfData = 2

class ClassFileBias(Enum):
    RandomAll = 1
    RandomPerClass = 2
    NumberOfPointsPerClass = 3
    NumberOfPointsOfClass = 4
    NumberOfPointsAll = 5

class Config:
    @staticmethod
    def IsLocalMachine():
        if os.path.isdir("C:/Program Files") or os.path.isdir("/home/jonas"):
            return True
        else: 
            return False
    
    @staticmethod
    def IsWindowsMachine():
        if os.path.isdir("C:/"):
            return True
        else: 
            return False

    def BatchSize(self):
        if Config.IsLocalMachine():
            size = 8
        else:
            size = 32
            
        return ceil(size / self.input_tile_count / (self.npoints / Config.npoints))

    #Placeholders
    classCount = Label.Semantic3D.Count-1
    classNames = Label.Semantic3D.Names[1:]
    validClasses = []
    class_weights = np.ones([Label.Semantic3D.Count])
    class_color = np.random.uniform(0, 1, size = [classCount, 3])

    validation_split = 0.3
    trainFiles = validationFiles = None
    testFiles = []
    excludeFiles = []
    Paths = Paths.Semantic3D
    dataSummaryFile = None
    testsAreFixed = True
    oneSamplePerFile = True
    BoostingSampler = None
    
    trainFileBias = ClassFileBias.NumberOfPointsPerClass
    valFileBias = ClassFileBias.RandomAll
    
    epochs = 100
    pointComponents = 3
    featureComponents = 3 #rgb    
    classCount = 0
    npoints = 8192
    # npoints = 100000
    blocksize = 8
    test_step = 0.5
    downsampling_voxel_size = 0.1
    name = ""
    storeFilesInRAM = False
    oneSamplePerFile = True
    dataFileExtension = ""
    rawDataExtension = ""
    
    classBalanceTrain = False
    classBalanceTest = False
    
    input_tile_count = 1

    #Algorithm configuration
    noFeature = False
    Fusion = False
    Scale = False
    Rotate = False
    Mirror = False
    Jitter = False
    FtrAugment = False

    ValidationInterval = 1 # validate model after each N epochs
    TestInterval  = None # test model after each N epochs
    ValidateOnOtherData = None
    
    TrainDataPipeline = DataPipeline.Sequence
    ValDataPipeline = DataPipeline.Sequence

    logsPath = "./logs"
    ### MODEL CONFIG
    pl = 64
    ### MODEL CONFIG

    def BuildSpecDict(self):
        return {
                "MultiScale": True if self.input_tile_count > 1 else False,
                "noFeature" : self.noFeature,
                "Fusion" : self.Fusion,
                "Scale" : self.Scale,
                "Rotate" : self.Rotate,
                "Mirror" : self.Mirror,
                "Jitter" : self.Jitter,
                "FtrAugment" : False if self.noFeature else self.FtrAugment,                
                }
        
    dataSummary = None
    datasets = []
    
    def __init__(self, skipInitialization):
        return
        
    def __init__(self, validationSplit = None, skipAdding = False, skipFileSearch = False):
        if not(validationSplit is None):
            self.validation_split = validationSplit
            
        if(skipFileSearch):
            return
        
        self.FindTrainAndValidationFiles()
        
        self.RAMfiles = {}
        files_to_store = []
        if(self.TrainDataPipeline == DataPipeline.Sequence):
            files_to_store.append(self.TrainFiles())
        if(self.ValDataPipeline == DataPipeline.Sequence):
            files_to_store.append(self.ValidationFiles())
            
        if(self.storeFilesInRAM and len(files_to_store) > 0):
            self.RAMFilesPointCount = {}
            for fileName in tqdm(np.unique(np.concatenate(files_to_store))):
                file = os.path.join(self.Paths.processedTrain, os.path.splitext(fileName)[0]+self.dataFileExtension)
                self.RAMfiles[file] = self.ReadFile(file)
                self.RAMFilesPointCount[fileName] = len(self.RAMfiles[file][0])
        
        if(not skipAdding):
            self.datasets.append(self)
    
    def FindTrainAndValidationFiles(self):
        import pandas as pd
        
        self.dataSummaryFile = self.Paths.pointCloudPath + '/summary.csv'
        if not(self.dataSummaryFile is None) and self.dataSummaryFile.endswith(".csv") and os.path.exists(self.dataSummaryFile):            
            self.dataSummary = pd.read_csv(self.dataSummaryFile)
            self.train_files_indices, self.validation_files_indices = self.SplitData(self.AllFiles())
        elif self.trainFiles is None and self.validationFiles is None:
            self.dataSummary = pd.DataFrame.from_dict({ "File" : Paths.GetFiles(self.Paths.processedTrain, onlyNames=True) })
            self.train_files_indices, self.validation_files_indices = self.SplitData(self.AllFiles())
        else:
            self.dataSummary = pd.DataFrame.from_dict({ "File" : self.trainFiles+self.validationFiles })
            self.train_files_indices = list(range(len(self.trainFiles)))
            self.validation_files_indices = list(range(len(self.trainFiles), len(self.trainFiles)+len(self.validationFiles)))
    
    def ReadFile(self, filePath):
        if(not filePath.endswith(self.dataFileExtension)):
            filePath += self.dataFileExtension
        
        if(filePath in self.RAMfiles):
            return self.RAMfiles[filePath]
        else:
            return ProcessFeatures(self, ReadXYZRGBLBL(filePath))
    
    def ConcatDataForTraining(self, config_type):
        cts = config_type(skipAdding = True)
        self.datasets.append(cts)
    
    def GetRandomDataset(self, validation = False):
        datasets = np.array([[i, len(dt.ValidationFiles()) if validation else len(dt.TrainFiles())] for i, dt in enumerate(self.datasets)])
        
        datasets = datasets[datasets[:, 1].astype(np.int64).argsort()] # sort files by point count
        datasets[:, 1] = datasets[:, 1].astype(np.int64).cumsum() # accumulate point count
        
        pt = np.random.randint(0, datasets[-1, 1], dtype=np.int64) if (len(datasets) > 1) else 0 #random points
        index = np.argmax(datasets[:, 1] > pt)
        return self.datasets[index]
    
    def ChooseRandomFileByNumberOfPoints(self, validation):
        
        configs = self.datasets if len(self.datasets) > 0 else [self]
        
        file_lists = [[i, data.ValidationFiles() if validation else data.TrainFiles() ] for i, data in enumerate(configs)]        
        files = []
        for i, file_list in file_lists:
            if(configs[i].storeFilesInRAM):
                files.extend([[i, file, configs[i].RAMFilesPointCount[file]] for file in file_list])
            else:
                files.extend([[i, file, int(self.dataSummary.loc[self.dataSummary['File'] == file]['Points'])] for file in file_list])
                
        files = np.array(files)
        files = files[files[:, 2].astype(np.int64).argsort()] # sort files by point count
        files[:, 2] = files[:, 2].astype(np.int64).cumsum() # accumulate point count
        
        pt = np.random.randint(0, files[-1, 2], dtype=np.int64) if (len(files) > 1) else 0 #random point
        index = np.argmax(files[:, 2].astype(np.int64) > pt)

        file = files[index]
        cts = configs[file[0].astype(int)]
        filePath = os.path.join(cts.Paths.processedTrain, str(file[1]))
        
        return filePath, cts
    
    def ChooseFileByMaxLoss(self, validation):
        
        configs = self.datasets if len(self.datasets) > 0 else [self]
        
        file_lists = [[i, data.ValidationFiles() if validation else data.TrainFiles() ] for i, data in enumerate(configs)]        
        files = []
        for i, file_list in file_lists:
            for file in file_list:
                file = os.path.join(configs[i].Paths.processedTrain, file)
                heatmap = configs[i].Samplers[file].record.numpy().reshape(-1)
                bias = heatmap[np.nonzero(heatmap)].mean()
                files.append([i, file, bias])
        
        files = np.array(files)
        
        # We require log probabilities tensorflow.random.categorical
        log_probs = tf.math.log([tf.reshape(files[:, 2].astype(np.float32), [-1])])
        # Generate a single random sample
        file_index = int(tf.random.categorical(log_probs, 1))
        
        file = files[file_index]
        cts = configs[file[0].astype(int)]
        
        return str(file[1]), cts
    
    def RotateClouds(self, validation):
        configs = self.datasets if len(self.datasets) > 0 else [self]
                
        for config in configs:
            files = config.ValidationFiles() if validation else config.TrainFiles()
            for file in files:
                file = os.path.join(self.Paths.processedTrain, file)
                if(file in config.RAMfiles):
                    pts, fts, lbl = config.RAMfiles[file]
                    pts = RotatePointCloud(pts)
                    config.RAMfiles[file] = (pts, fts, lbl)
    
    def GetData(self, validation):
        
        if(not (self.BoostingSampler is None) and not validation):
            filePath, cts = self.ChooseFileByMaxLoss(validation)
        else:
            filePath, cts = self.ChooseRandomFileByNumberOfPoints(validation)
                
        (pts, fts, lbs) = cts.ReadFile(filePath)
        return pts, fts, lbs
    
    def GetClassFile(self, label : int, validation = False):
        """
        Input: class number
        Output: sample file that contains this class
        """
        file_indices = self.validation_files_indices if validation else self.train_files_indices
        
        if(len(self.dataSummary.columns) == 1):
            allFiles = np.array(self.dataSummary['File'])[file_indices]
            rand = np.random.randint(len(allFiles))
            fileName = allFiles[rand]
        elif(label != -1):
            files = np.array(self.dataSummary[['File', str(self.validClasses[label])]])[file_indices]
            # files = np.array(self.dataSummary[['File', str(self.validClasses[label])]])[self.train_files_indices]
                    
            files = files[files[:, 1].astype(int).argsort()] # sort files by point count
            files[:, 1] = files[:, 1].astype(int).cumsum() # accumulate point count
            
            pt = np.random.randint(0, files[-1, 1], dtype=np.int64) if (len(files) > 1) else 0 #random points
            index = np.argmax(files[:, 1] > pt)
            fileName = files[index, 0]
        else:
            # find files that contain all classes
            idx = np.where(np.sum(np.array(self.dataSummary[[str(cls) for cls in self.validClasses]])[file_indices] > 0, -1) == len(fileClasses))
            fileName = self.dataSummary.loc[file_indices[idx[np.random.randint(0, len(idx))]][0]]['File']        
        
        return os.path.join(self.Paths.processedTrain, fileName)
    
    def DataPath(self):
        return self.Paths.processedTrain
    
    def RawDataPath(self):
        return self.Paths.rawTrain

    def AllFiles(self):        
        return np.array(self.dataSummary['File'])
    
    def TrainFiles(self):
        return np.array(self.dataSummary['File'])[self.train_files_indices]

    def ValidationFiles(self):
        return np.array(self.dataSummary['File'])[self.validation_files_indices]
    
    def TestFiles(self):
        return self.testFiles

    def FileIndices(self, files):        
        files_names = tuple([os.path.splitext(os.path.basename(file))[0] for file in files])
        return np.array([i for i, file in enumerate(self.AllFiles()) if file.startswith(files_names)])

    def SplitData(self, files):
        # np.random.seed(0) #reproduce same random choice for files
        
        train_files_indices = np.array([i for i in range(len(files)) 
                               if os.path.exists(os.path.join(self.Paths.processedTrain, files[i]+self.dataFileExtension))
                               and not (os.path.splitext(os.path.basename(files[i]))[0] in self.testFiles)])
        
        random_indices = np.random.choice(range(len(train_files_indices)), int(len(train_files_indices)*self.validation_split), replace=False).astype(int)        
        validation_files_indices = train_files_indices[random_indices]
        train_files_indices = np.delete(train_files_indices, random_indices, axis=0)

        return train_files_indices, validation_files_indices

    def Name(self, UID = ""):
        modelName = self.name
        
        trainfiles = np.sum([len(cts.TrainFiles()) for cts in self.datasets]) if (len(self.datasets) > 0) else len(self.TrainFiles())
        validationfiles = np.sum([len(cts.ValidationFiles()) for cts in self.datasets]) if (len(self.datasets) > 0) else len(self.ValidationFiles())
        modelName += f"({trainfiles}-{validationfiles}-{len(self.TestFiles())})"

        for spec, value in self.BuildSpecDict().items():
            if(value == True):
                modelName += f"({spec})"

        if(UID != ""):
            modelName += f"_{UID}"

        return modelName
    
    def NormalizeData(self, points, features = None, labels = None, validation = False):
        if(labels is None):
            return points, features, labels
        
        if(not validation):
            mask = np.where(labels != 0)[0]
            if(len(mask) > 0):
                points = points[mask]
                labels = labels[mask]

                if(not features is None):
                    features = features[mask]
        
        # labels = self.MapLabels(labels, mapLabels)
        labels -= 1

        return points, features, labels
    
    def RevertLabels(self, labels):
        return labels

    def MapLabels(self, labels, type):
        return labels
    
    def RevertData(self, points, features = None, labels = None):
        return points, features, labels
    
    def GetMetaInfo(self):
        feature = 'none'
        if not self.noFeature and self.featureComponents == 1:
            feature = 'intensity'
        elif not self.noFeature and self.featureComponents == 3:
            feature = 'RGB'

        input_features_tensor = "" if feature == 'none' else "serving_default_input_2:0"

        classOutputs = {clsName: [i, i] for i, clsName in enumerate(self.classNames)}

        metaInfo = {
            "feature": feature,
            "input_points_tensor": "serving_default_input_1:0",
            "input_features_tensor": input_features_tensor,
            "output_tensor": "StatefulPartitionedCall:0",
            "tile_size": float(self.blocksize),
            "tile_step": float(self.test_step),
            "downsampling_voxel_size": float(self.downsampling_voxel_size),
            "class_names": self.classNames,
            "class_outputs": classOutputs
        }

        return metaInfo

    @staticmethod
    def RemoveUID(name : str):
        return name.replace(f"_{ParseModelUID(name)}", "")
    
    @staticmethod
    def UID():
        import uuid
        return uuid.uuid4().hex[:10]

    # def TestFiles(self):        
    #     return Paths.JoinPaths(self.Paths.processedTrain, self.testFiles)

    # def TrainFiles(self):
    #     return Paths.GetFiles(self.Paths.processedTrain, excludeFiles = self.TestFiles()+self.excludeFiles)
    
    # def ValidationFiles(self):
    #     return []        
        
    def GetClassColors(self):
        return np.random.uniform(0, 1, size = [self.classCount, 3])
    
    def GetDataPipeline(self, train_pipeline = True, batch_count = None):
        return None

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

def ParseModelUID(file):
        parts = Paths.FileName(file).split("_")

        if(len(parts) >= 2):
            return parts[1]
        else:
            return None

def ParseModelName(file, withUID = True):
    parts = Paths.FileName(file, withoutExt = False).split("_")

    name = parts[0]
    if(withUID and len(parts) > 1):
        name += "_"+parts[1]

    return name