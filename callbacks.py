import tensorflow as tf
from models import SaveModel
import os, shutil
import wandb
from configs.Config import Config
import numpy as np
from data import TestSequence, TrainSequence
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from metrics import stats_iou_per_class
from wandb.keras import WandbCallback
from data import GetValidationData, RandomChoice
from time import time
from functools import partial
from multiprocessing import Pool

logSaveDir = "./"
logFileName = "run.log"

def WriteToLog(msg):
    if(os.path.isdir(logSaveDir)):
        logFile = open(logSaveDir+logFileName, "a")
        logFile.write(msg+"\n")
        logFile.close()

def PrintToLog(msg):
    print(msg)
    WriteToLog(msg)

class IOUPerClass(tf.keras.callbacks.Callback):
    def __init__(self, plot_path, classNames, firstEpoch = 0, metric = "miou"):
        self.metric = metric
        self.epoch = firstEpoch    
        self.classCount = len(classNames)
        self.classNames = classNames
        self.path = plot_path

        print(f"IOU logs path: {self.path}")

        self.writers = {}
        self.val_writers = {}
        ioupath = os.path.join(plot_path, "iou")
        os.makedirs(ioupath, exist_ok=True)
        for i in range(self.classCount):
            path = os.path.join(ioupath, classNames[i])
            os.makedirs(path, exist_ok=True)
            self.writers[classNames[i]] = tf.summary.create_file_writer(path)

            path = os.path.join(ioupath, "val_"+classNames[i])
            os.makedirs(path, exist_ok=True)
            self.val_writers["val_"+classNames[i]] = tf.summary.create_file_writer(path)
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
            tf.summary.scalar(tag, value[0], step=epoch)
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
    
    def WriteToFile(self, logs, metric_prefix=""):
        # metrix = logs.get(metric)
        # iou_score = [i[0] for i in metrix[len(metrix)-self.classCount:]]
        iou_score = []
        for className in self.classNames:
            metrix = logs.get(metric_prefix + className)
            iou_score.append(metrix[0])

        PrintToLog(f"IOU: " + "".join([f"{name}({score*100:.1f}%) " for name, score in zip(self.classNames, iou_score)]))
    
    def on_epoch_end(self, batch, logs=None):
        # self.WriteLogs(self.writers, self.metric, logs, self.epoch)
        # self.WriteLogs(self.val_writers, "val_"+self.metric, logs, self.epoch)

        for className in self.classNames:
            self.WriteLog(self.writers[className], className, logs, self.epoch)
            self.WriteLog(self.val_writers["val_" + className], "val_" + className, logs, self.epoch)

        self.WriteLog(self.miou_writer, self.metric, logs, self.epoch)
        self.WriteLog(self.val_miou_writer, "val_"+self.metric, logs, self.epoch)

        self.WriteToFile(logs, metric_prefix="val_")       

        self.epoch += 1
        
class ModelSaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, saveDir, trainingSteps, metric = "accuracy", val_metric_prefix = "val_", modelNamePrefix = "", classNames = [], sendNotifications = False, saveOnlyBestModel = False, metaInfo = None):
        super().__init__()
        self.saveDir = saveDir
        self.metric = metric
        self.val_metric = val_metric_prefix+self.metric
        self.modelNamePrefix = modelNamePrefix
        self.classNames = classNames
        self.best_val_miou = 0
        self.saveOnlyBestModel = saveOnlyBestModel
        self.metaInfo = metaInfo

        self.epoch = 0
        self.trainingSteps = trainingSteps
        
        self.sendNotifications = sendNotifications
        if(self.sendNotifications):
            self.notifyDevice = Notify()
        
        os.makedirs(self.saveDir, exist_ok=True)
        WriteToLog(f"Training: {modelNamePrefix}")
        WriteToLog(f"Model saving tracks '{self.val_metric}' metric.")
    
    def delete_tf_model_saves(self):
        for x in os.listdir(self.saveDir):
            path = os.path.join(self.saveDir, x)
            if(os.path.isdir(path)):
                shutil.rmtree(path, ignore_errors=False, onerror=None)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch + 1
        if(len(logs) > 0):
            miou = float(logs.get(self.metric)*100)             
            val_miou = float(logs.get(self.val_metric)*100)
            
            # saves only the best model in TF format
            saveTFformat = True
            if(self.saveOnlyBestModel):
                if(val_miou < self.best_val_miou):                    
                    saveTFformat = False
                else:
                    self.best_val_miou = val_miou
                    self.delete_tf_model_saves()
                
            SaveModel(self.saveDir, epoch, self.model, miou, val_miou, self.modelNamePrefix, saveTFFormat = saveTFformat, metaInfo = self.metaInfo)

            message = "Ep: {0}. {1}: {2:.3}%. {3}: {4:.3}%".format(self.epoch, self.metric, miou, self.val_metric, val_miou)
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
    
class LogToWandb(tf.keras.callbacks.Callback):
    def __init__(self, samples, validationInterval, prefix, config : Config, mapConfig = None):
        self.samples = samples
        self.validationInterval = validationInterval
        self.config = config
        self.mapConfig = mapConfig
        self.prefix = prefix
    
    def on_epoch_end(self, epoch, logs=None):
        if(epoch % self.validationInterval != 0):
            return
        
        print(f"LogToWandb {self.prefix}")
        
        clsAll = self.model.predict(self.samples[0], batch_size = self.config.BatchSize(), verbose = 1)
        clsAll = np.expand_dims(clsAll.argmax(-1), -1)
        
        if(not self.mapConfig is None):            
            clsAll = np.stack([self.config.MapLabels(cls+1, self.mapConfig.name) for cls in clsAll])

        for i in range(len(clsAll)):
            if(len(self.samples[0]) == 2):                
                intensity = self.samples[0][0][i]
                colors = np.tile(intensity, [1, 3])
                if(colors.max() <= 1):
                    colors *= 255
                pts = self.samples[0][1][i]
            else:
                colors = None
                pts = self.samples[0][i]

            pred_cls = clsAll[i]
            true_cls = np.expand_dims(self.samples[1][i].argmax(-1), -1)

            # true_other = np.where(true_cls == 0)[0]
            # true_lanes = np.where(true_cls == 1)[0]
            # pred_other = np.where(pred_cls == 0)[0]
            # pred_lanes = np.where(pred_cls == 1)[0]

            # colors = np.full((len(true_cls), 3), [255,255,255])            
            # colors[np.where(np.logical_and(true_cls == 0, pred_cls == 0))[0]] = [0,0,255]            
            # colors[np.where(np.logical_and(true_cls == 1, pred_cls == 0))[0]] = [255,255,0]
            # colors[np.where(np.logical_and(true_cls == 1, pred_cls == 1))[0]] = [0,255,0]
            # colors[np.where(np.logical_and(true_cls == 0, pred_cls == 1))[0]] = [255,0,0]
            # pts = np.concatenate([pts, colors], axis=-1)
            
            pts = pts.reshape([-1, pts.shape[-1]])
            if not (colors is None):
                colors = colors.reshape([-1, colors.shape[-1]])
            true_cls = true_cls.reshape([-1, true_cls.shape[-1]]).astype(np.float32) #+ 1 # WANDB classification is from [1 to 14]
            pred_cls = pred_cls.reshape([-1, pred_cls.shape[-1]]).astype(np.float32) #+ 1 # WANDB classification is from [1 to 14]
                        
            classColors = self.config.GetClassColors() if (self.mapConfig is None) else self.mapConfig.GetClassColors()
            true_cls_color = classColors[true_cls.flatten().astype(int)]*255
            pred_cls_color = classColors[pred_cls.flatten().astype(int)]*255
            
            pointclouds = []
            if(not colors is None):
                pointclouds.append(wandb.Object3D({"type": "lidar/beta","points": np.concatenate([pts, colors], axis=-1),}))                    
            pointclouds.append(wandb.Object3D({"type": "lidar/beta","points": np.concatenate([pts, true_cls_color], axis=-1),}))
            pointclouds.append(wandb.Object3D({"type": "lidar/beta","points": np.concatenate([pts, pred_cls_color], axis=-1),}))

            wandb.log(
            {
                # f"point_scene_{i+1}": wandb.Object3D({"type": "lidar/beta","points": pts,}),
                f"{self.prefix}_classification_scene_{i+1}": pointclouds,
            })
            
class Validation(tf.keras.callbacks.Callback):
    def __init__(self, data_pipeline, testsAreFixed : bool, validationInterval, config : Config, prefix = "", mapConfig = None):
        self.data_pipeline = data_pipeline
        self.validationInterval = validationInterval
        self.config = config
        self.prefix = prefix
        self.testsAreFixed = testsAreFixed
        self.mapConfig = mapConfig
        
        if(testsAreFixed):
            self.data_pipeline = GetValidationData(self.data_pipeline)
    
    def on_epoch_end(self, epoch, logs=None):
        if(epoch % self.validationInterval != 0):
            return    
        
        print(f"{self.prefix} Validation")
        
        classNames = []
        if(self.mapConfig is None):
            if(isinstance(self.data_pipeline, tuple)):
                results = self.model.evaluate(self.data_pipeline[0], self.data_pipeline[1], batch_size = self.config.BatchSize(), workers = 4, verbose = 1)                
            else:
                results = self.model.evaluate(self.data_pipeline, batch_size = self.config.BatchSize(), workers = 4, verbose = 1)                
            
            loss = results[0]
            miou = results[1][0][0]
            iou = results[1][1:].flatten()
            classNames = self.config.classNames
        else:
            pred_cls = self.model.predict(self.data_pipeline, batch_size = self.config.BatchSize(), verbose = 1)
            true_cls = self.data_pipeline[1].argmax(-1).flatten()
            pred_cls = self.config.MapLabels(pred_cls.argmax(-1).flatten()+1, self.mapConfig.name)
            
            cm = confusion_matrix(true_cls, pred_cls, labels=list(range(self.mapConfig.classCount)))
            results = stats_iou_per_class(cm)
            
            loss = None
            miou = results[0]
            iou = results[1]
            classNames = self.mapConfig.classNames
        
        logs[f"{self.prefix}_val_loss"] = loss
        logs[f"{self.prefix}_val_miou"] = miou
        
        for i, val in enumerate(iou):
            logs[f"{self.prefix}_val_{classNames[i]}"] = val
            
class ProcessLog(tf.keras.callbacks.Callback): 
    def __init__(self, iou_classes):
        self.iou_classes = iou_classes
              
    def on_batch_end(self, batch, logs=None):
        logs = self.update_log(logs)
        
    def on_epoch_end(self, batch, logs=None):
        logs = self.update_log(logs)
        
    def update_log(self, logs):
        logs = self.split_iou(logs, "")
        logs = self.split_iou(logs, "val_")
        return logs
    
    def split_iou(self, logs, prefix = ""):
        miou = prefix+'miou'
        
        if(miou in logs):
            logs.update({prefix+cls:val[0] for val, cls in zip(logs[miou][1:], self.iou_classes)})
            logs[miou] = logs[miou][0][0]
            
        return logs        
    
class LogToWandbAfterEachBatch(tf.keras.callbacks.Callback):              
    def on_batch_end(self, batch, logs=None):
        wandb.log(logs, commit=True)

class LogToWandbLargeScale(tf.keras.callbacks.Callback):
    def __init__(self, files, consts : Config, upload_pointclouds = True):
        self.files = files 
        self.consts = consts
        self.testInterval = consts.TestInterval
        self.upload_pointclouds = upload_pointclouds
        
        # self.data = [self.PreComputeFile(file, consts) for file in tqdm(files)]
        
        compute_func = partial(self.PreComputeFile, consts = consts)
        with Pool(8) as p:
            self.data = list(tqdm(p.imap(compute_func, files), total=len(files)))
            
        print("Done precomputing")
    
    def PreComputeFile(self, file, consts):
        datasetName = os.path.basename(file)
        datasetName = os.path.splitext(datasetName)[0]

        seq = TestSequence(file, consts, localMachineCap=False, generateALl = True)

        xyz = seq.xyzrgblbl[:,:3]
        
        # blockSize = self.consts.blocksize*(sqrt(self.consts.input_tile_count))
        # DataTool().VisualizePointCloudAsync([xyz], bBoxes = [BoundingBoxFromVoxel([pt[0], pt[1], min(xyz[:,2])+(blockSize/2)], blockSize) for pt in seq.allpts])

        if(not consts.noFeature):
            if(consts.featureComponents == 1):
                ptColors = np.tile(seq.xyzrgblbl[:,3,np.newaxis], [1, 3])
            elif(consts.featureComponents == 3):
                ptColors = seq.xyzrgblbl[:,3:6]
        else:
            ptColors = None

        true_labels = seq.lbl

        batches = seq.__getitem__(0) # will generate all samples at once        

        idx = seq.idxList

        return [datasetName, xyz, ptColors, true_labels, batches, idx]        
    
    def on_epoch_end(self, epoch, logs=None, fix_labels_func = None):
        if(epoch % self.testInterval != 0):
            return

        ConfusionMatrix = None
        # confidences = []
        
        print(f"{self.consts.name} Large Scale Test")
        
        for [datasetName, xyz, ptColors, true_labels, batches, idx] in self.data:
            
            scores = np.zeros((len(xyz), self.consts.classCount))
            observations = np.zeros((len(xyz)))
            t = time()
            output = self.model.predict(batches, batch_size = self.consts.BatchSize(), verbose = 1)            
            
            idx = idx.reshape([-1])
            output = output.reshape([-1, output.shape[-1]])
                
            # for i in range(len(output)):
            #     scores[idx[i]] += output[i]
            scores[idx] += output
            observations[idx] += 1
            
            if(not fix_labels_func is None):
                scores = fix_labels_func(xyz, scores)

            # remove not observed points
            observed_points = np.logical_not(scores.sum(1)==0).flatten()
            # remove not ground truth points with label 0
            labeled_points = np.logical_not(true_labels==-1).flatten()
            mask = np.where(np.logical_and(observed_points, labeled_points))[0]            
            scores = scores[mask]
            pred_labels = np.expand_dims(scores.argmax(1), -1)
            
            # Calculate accuracy metrics
            cm = confusion_matrix(true_labels[mask], pred_labels, labels=list(range(self.consts.classCount)))
            ConfusionMatrix = cm if (ConfusionMatrix is None) else ConfusionMatrix + cm            
            iou = stats_iou_per_class(cm)
            miou = iou[0]
            logs[f"{datasetName}_MIOU"] = miou            
            for i, val in enumerate(iou[1]):
                logs[f"{datasetName}_iou_{self.consts.classNames[i]}"] = val
            
            # logs[f"{datasetName}_confidence"] = np.mean(confidence)
            PrintToLog(f"Test segmentation '{datasetName}': {len(batches[0])} tiles processed in {time()-t:.1f} sec. MIOU: {miou*100:.2f}%")

            if(self.upload_pointclouds):
                # TRIM POINTS TO NOT DESTROY WANDB
                # IT SUPPORTS UP TO 300,000 points            
                uploadPoints = 300000
                if(len(mask) > uploadPoints):
                    choice = RandomChoice(len(mask), uploadPoints, replace=False)
                else:
                    choice = np.arange(0, len(mask), dtype=int)

                temp_xyz = xyz[mask][choice]

                # confidenceColor = np.tile([[1.,1.,0]], [len(temp_xyz), 1])
                # confidence = np.expand_dims(scores[choice].max(-1) / observations[mask][choice], axis=-1)
                # confidences.append(np.mean(confidence))
                # confidence_normalized = (confidence - confidence.min()) / (confidence.max() - confidence.min())
                # confidenceColor += np.tile([[-1.,0,0]], [len(temp_xyz), 1]) * confidence_normalized            
                # confidenceColor[np.where(true_labels[mask][choice] != pred_labels[choice])[0]] = np.array([1., 0, 0])
                # confidenceColor *= 255

                pointclouds = []
                # if not(ptColors is None):
                #     pointclouds.append(wandb.Object3D({"type": "lidar/beta","points": np.concatenate([temp_xyz, ptColors[mask][choice]], axis=-1),}))
                # low confidence points - yellow, high confidence - green, not accurate points - red
                # pointclouds.append(wandb.Object3D({"type": "lidar/beta","points": np.concatenate([temp_xyz, confidenceColor], axis=-1),}))
                # pointclouds.append(wandb.Object3D({"type": "lidar/beta","points": np.concatenate([temp_xyz, true_labels[mask][choice]], axis=-1),}))

                classColors = self.consts.GetClassColors()
                pred_cls_color = classColors[pred_labels[choice].flatten().astype(int)]*255
                pointclouds.append(wandb.Object3D({"type": "lidar/beta","points": np.concatenate([temp_xyz, pred_cls_color], axis=-1),}))

                wandb.log({ f"Large scale samples/{datasetName}": pointclouds })
                
        iou = stats_iou_per_class(ConfusionMatrix)
        # logs["test_confidence"] = np.mean(confidences)
        logs["test_miou"] = iou[0]
        for i, val in enumerate(iou[1]):
            logs[f"test_iou_{self.consts.classNames[i]}"] = val

        PrintToLog(f"Test miou: {iou[0]*100:.2f}%")