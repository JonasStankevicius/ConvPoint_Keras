from dataTool import ReadXYZL, UpscaleToOriginal, SplitFilesIntoChunks, DataTool
from imports import *
import math
import numpy as np
from tqdm import tqdm

import wandb
from wandb.keras import WandbCallback

import tensorflow as tf

from sklearn.metrics import confusion_matrix
import metrics

# from notify_run import Notify

from models import *
from data import *
from configs.Config import Config, DataPipeline
from configs.Semantic3D import Semantic3D
from configs.Aerial import Aerial
from configs.Sem3DExtended import Sem3DExtended
from configs.SDE import SDE
from configs.NMG import NMG
from callbacks import *

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
    output = model.predict(seq, workers = consts.BatchSize(), max_queue_size = 300, verbose = 1)

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
        
def TrainModel(config : Config, trainSteps = 100, testSteps = 30, modelPath = None, saveDir = Paths.dataPath, first_epoch = 0, epochs = None, sendNotifications = False, CUSTOM_LOOP = False):
    model = None
    modelName = None
    if(modelPath != None):
        if(not isinstance(modelPath, list)):
            modelName = Config.ParseModelName(modelPath)
            if(config.Name() != Config.RemoveUID(modelName)):
                modelName = config.Name(config.UID())
        logSaveDir = saveDir + f"/{modelName}/"

        if(isinstance(modelPath, list)):
            model = FuseModels(modelPath, config)
        else:
            model, modified = LoadModel(modelPath, config)
            if(not modified):
                first_epoch = ParseEpoch(modelPath) +1
    else:
        if(config.Fusion):
            model = FuseModels(None, config)
        else:
            if(config.input_tile_count == 1):
                model = CreateModel(config)
            else:
                model = CreateTwoScaleModel(config)
                # model = CreateTwoScaleModelPretrained(pretrained, consts)
    
    # model.summary(print_fn=lambda x: open('modelsummary.txt', 'a').write(x + '\n'))
        
    if(modelName is None or modelName == ""):
        modelName = config.Name(config.UID())
        logSaveDir = saveDir + f"/{modelName}/"
    logsPath = os.path.join(config.logsPath, Config.RemoveUID(modelName))
    os.makedirs(logsPath, exist_ok=True)

    trainingSteps = int((trainSteps*16)/(config.BatchSize()*config.input_tile_count)) if not Config.IsLocalMachine() else int(30)
    validationSteps = int((testSteps*16)/(config.BatchSize()*config.input_tile_count)) if not Config.IsLocalMachine() else int(10)

    PrintToLog(f"Model: {modelName}")
    PrintToLog(f"Train on {len(config.trainFiles)} : {config.trainFiles}")
    PrintToLog(f"Validate on {len(config.validationFiles)} : {config.validationFiles}")
    PrintToLog(f"Test on {len(config.testFiles)} : {config.testFiles}")
    PrintToLog(f"Batch size: {config.BatchSize()}, train batches: {trainingSteps}, validation batches: {validationSteps}")
    
    callbacks_list = [
        ProcessLog(config.classNames),
        # IOUPerClass(logsPath, consts.classNames, first_epoch+1),        
        # tf.keras.callbacks.TensorBoard(log_dir=logsPath, histogram_freq = 1, profile_batch = 0) # to disable profiling profile_batch = 0, to enable profile_batch = n        
    ]
        
    if config.TrainDataPipeline == DataPipeline.Sequence:
        train_data_pipeline = TrainSequence(trainingSteps, config, randomPoints = True, balanceClasses = config.classBalanceTrain, validation = False)
    else:
        train_data_pipeline = config.GetDataPipeline(True, trainingSteps)
        
    if config.ValDataPipeline == DataPipeline.Sequence:
        val_data_pipeline = TrainSequence(validationSteps, config, randomPoints = False, dataAugmentation = False, balanceClasses = config.classBalanceTest, validation = True)
    else:        
        val_data_pipeline = config.GetDataPipeline(False, validationSteps)
        
    if(not(config.ValidationInterval is None)):# and not Config.IsLocalMachine()):
        # validationData.balanceClasses = True
        # testSamples = GetValidationData(validationData, config, batchSize = 10, batchCount = 1)
        # validationData.balanceClasses = config.classBalanceTest
        # callbacks_list.append(LogToWandb(testSamples, config.ValidationInterval, config.name, config))
        callbacks_list.append(Validation(val_data_pipeline, config.testsAreFixed, config.ValidationInterval, config, prefix = config.name))
    
    if(not (config.ValidateOnOtherData is None)):
        if(config.ValidateOnOtherData == "Semantic3D"):
            temp_cfg = config.datasets[1]
            assert isinstance(temp_cfg, Semantic3D)
                
            temp_cfg.classCount = config.classCount
            temp_cfg.class_color = config.class_color
            temp_cfg.noFeature = config.noFeature
            temp_cfg.classNames = config.classNames            
        elif(config.ValidateOnOtherData == "SDE"):
            SDE.validation_split = 1.0
            temp_cfg = SDE(skipAdding = True)
        else:
            assert "Not Supported validation data"
                
        data = TrainSequence(validationSteps*0.5, temp_cfg, randomPoints = False, dataAugmentation = False, 
                                balanceClasses = config.classBalanceTest, validation = True)
        callbacks_list.append(Validation(data, temp_cfg.testsAreFixed, config.ValidationInterval, config, prefix = temp_cfg.name, mapConfig = temp_cfg))
        
        data.balanceClasses = True
        testSamples = GetValidationData(data, temp_cfg, batchSize = 9, batchCount = 1)
        data.balanceClasses = config.classBalanceTest
        callbacks_list.append(LogToWandb(testSamples, config.ValidationInterval, temp_cfg.name, config, mapConfig = temp_cfg))

    if(not(config.TestInterval is None)):# and not Config.IsLocalMachine()):        
        files = [os.path.join(config.Paths.processedTrain, os.path.splitext(fileName)[0]+config.rawDataExtension) for fileName in config.TestFiles()]
        files = [file for file in files if os.path.exists(file)]
        if(len(files) > 0):
            callbacks_list.append(LogToWandbLargeScale(files, consts, upload_pointclouds = False))
        else:
            print("No test files found!")
    
    if(epochs is None):
        epochs = 20 if config.Fusion else 100
        
    # if(not Config.IsLocalMachine()):
    wandb.config.weights = model.count_params() #if (consts.input_tile_count == 1) else (downSamplingModel.count_params() + largeScaleModel.count_params() + upSamplingModel.count_params())    
    wandb.config.trainingSteps = trainingSteps
    wandb.config.validationSteps = validationSteps
    wandb.run.name = modelName
    # wandb.run.save()
    # Add wandb last to log all metrcis pushed by other callbacks
    callbacks_list += [LogToWandbAfterEachBatch(), WandbCallback(save_model = False)]
    
    # Add model saving last to get all validation values
    if(not Config.IsLocalMachine()):
        callbacks_list.append(ModelSaveCallback(logSaveDir, trainingSteps, "miou", val_metric_prefix = f"{config.name}_val_", modelNamePrefix = modelName, sendNotifications=sendNotifications, saveOnlyBestModel = True, metaInfo = config.GetMetaInfo()))
    
    
    if(CUSTOM_LOOP):
        TrainLoop(consts, model, train_data_pipeline, epochs, callbacks_list)
    else:
        model.fit(train_data_pipeline, epochs = epochs, batch_size = config.BatchSize(), initial_epoch = first_epoch, callbacks = callbacks_list, workers = 1, use_multiprocessing = False, max_queue_size = 0)        

def TrainLoop(consts, model, trainData : TrainSequence, epochs, callbacks):
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, epsilon = 1e-8)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    loss_fn_not_reduced = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    
    train_acc_metric = metrics.MIOU(7)
    
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss_value = loss_fn(y, logits)
        loss_per_sample = tf.math.reduce_sum(loss_fn_not_reduced(y, logits), -1) / tf.cast(tf.shape(y)[1], tf.float32)        
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_acc_metric.update_state(y, logits)
        
        return loss_value, loss_per_sample
        
    for epoch in range(epochs):
        # Iterate over the batches of the dataset.
        pbar = tqdm(range(len(trainData)), total=len(trainData))            
        loss_metric = []
                        
        for step in pbar:            
            x_batch_train, y_batch_train = trainData.__getitem__(-1)
            loss_value, loss_per_sample = train_step(x_batch_train, y_batch_train)
            trainData.set_loss_values(loss_per_sample.numpy())
            # DataTool().VisualizePointCloudAsync([x_batch_train[7]], classes = [y_batch_train[7]])
            trainData.prefecth()
            loss_metric.append(loss_value)
            wandb.log({
                "loss":np.array(loss_metric).mean(),
                "miou":float(train_acc_metric.result()[0]),
            })
            pbar.set_description(f"{epoch} Epoch | Step:{step}")
        
        logs = {
            "loss":np.array(loss_metric).mean(),
            "miou":train_acc_metric.result().numpy(),
        }
        for callback in callbacks:
            callback.model = model
            callback.on_epoch_end(epoch, logs)
        train_acc_metric.reset_state()
        
        wandb.log(logs)
        print(f"Val_Loss: {logs['Sem3DExt_val_loss']:.3f}, Val_Miou: {logs['Sem3DExt_val_miou']:.3f}")

def EvaluateModels(modelsList, testFiles, consts, x = None, y = None):
    if(x is None or y is None):
        validationSteps = int(((150 if not Config.IsLocalMachine() else 10) * 16)/consts.BatchSize())
        x, y = GetValidationData(testFiles, consts, validationSteps, newDataGeneration = False)

    for file in modelsList:
        model, _ = LoadModel(file, consts)
        metrics = model.evaluate(x, y, batch_size = consts.BatchSize(), workers = consts.BatchSize(), max_queue_size = 300)
        # print(f"miou: {metrics[2][0][0]*100:.3}")

def GenerateLargeFile(model, consts, outputFile, originalFile = None, Upscale = True, saveScores = True):
    from dataTool import ReadXYZ
    from tqdm import tqdm

    seq = TestSequence(originalFile, consts, splitDataSetToParts=16000)
    print("All pts: ", len(seq.allpts))

    xyzrgb = seq.xyzrgblbl[:,:3]
    scores = np.zeros((xyzrgb.shape[0], consts.classCount))

    for _ in tqdm(range(seq.LenParts())):
        seq.NextPart()
        output = model.predict(seq, workers = 8, max_queue_size = 300, verbose = 1)

        idx = seq.idxList
        for i in range(len(output)):
            scores[idx[i]] += output[i]

    # remove not observed points
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

def MeasureInference(model, file, consts : Config, tile_step, original_points, original_labels, saveLabels = False, modelName = None, labelFixFunc = None):
    from tqdm import tqdm
    from metrics import stats_iou_per_class, stats_overall_accuracy

    consts.test_step = tile_step
    seq = TestSequence(file, consts, splitDataSetToParts=5000, localMachineCap=False, generateALl = True)

    xyz = seq.xyzrgblbl[:,:3]
    scores = np.zeros((len(xyz), consts.classCount))
    observed_points = np.zeros((len(xyz)), np.int)

    logsPath = os.path.join(consts.logsPath, consts.Name())
    os.makedirs(logsPath, exist_ok=True)

    t = time()

    executionTime = 0
    for i in tqdm(range(seq.LenParts())):
        seq.NextPart()
        
        pts = seq.__getitem__(i)
        PrintToLog(f"Average number of points in tile {i+1}: {seq.avg_pts_in_tile}.")

        t = time()
        output = model.predict(pts, batch_size = 16, workers = 8, max_queue_size = 300, verbose = 1)
        # output = model.predict(pts, workers = 8, max_queue_size = 300, verbose = 1)
        executionTime += time() - t

        idx = seq.idxList
        for i in range(len(output)):
            scores[idx[i]] += output[i]
            observed_points[idx[i]] += 1
    t = time() - t

    if not(labelFixFunc is None):
        scores = labelFixFunc(xyz, scores)
    
    # remove not observed points
    mask = np.logical_not(scores.sum(1)==0)
    scores = scores[mask]
    classifieldpts = xyz[mask]
    # classifieldpts, _, _ = consts.RevertData(classifieldpts)
    pred_labels = scores.argmax(1)

    if not(original_points is None):
        pred_labels = UpscaleToOriginal(original_points, classifieldpts[:,:3], pred_labels).astype(int)
        
        classified = np.where(original_labels != -1)[0]
        cm = confusion_matrix(original_labels[classified], pred_labels[classified])
        iou_score = stats_iou_per_class(cm)[1]
        miou_score = np.mean(iou_score)
        acc_score = stats_overall_accuracy(cm)

        if(saveLabels):
            las_name = os.path.splitext(file)[0]+f"_{'' if (modelName is None) else modelName}_step_{tile_step}_upscaled_lbl.las"
            SaveToLas(las_name, original_points, labels = np.expand_dims(pred_labels, 1))
    else:
        pred_labels = UpscaleToOriginal(xyz[:,:3], classifieldpts[:,:3], pred_labels).astype(int)
        
        classified = np.where(seq.lbl != -1)[0]                
        cm = confusion_matrix(seq.lbl[classified], pred_labels[classified])
        iou_score = stats_iou_per_class(cm)[1]
        miou_score = np.mean(iou_score)
        acc_score = stats_overall_accuracy(cm)

        if(saveLabels):
            las_name = os.path.splitext(file)[0]+f"_{'' if (modelName is None) else modelName}_step_{tile_step}_voxels_lbl.las"
            SaveToLas(las_name, xyz[:,:3], labels = pred_labels)

    PrintToLog(f"Model name: {modelName}")
    PrintToLog(f"Step: {tile_step}. Raw points: {0 if original_points is None else len(original_points)}. Voxelized Points: {len(xyz)}. Classified points: {len(classifieldpts)}. Tile size: {consts.blocksize}. Points in tile: {consts.npoints}")
    PrintToLog(f"Classified points: {(len(classifieldpts) / len(xyz))*100:.1f}%. Each point observed around {np.mean(observed_points):.1f} times.")
    PrintToLog(f"Tiles: {len(seq.allpts)}. Model inference time: {executionTime:.3f} sec.")   
    PrintToLog(f"MIOU: {miou_score*100:.1f}%. Accuracy: {acc_score*100:.1f}%")    
    PrintToLog(f"IOU: " + "".join([f"{name}({score*100:.1f}%) " for name, score in zip(consts.classNames, iou_score)]))
    PrintToLog("---------------------------------------------------------------")


# logFileName = "real_model_sliding_window_upscale_to_raw.log"

def SlidingOverlapTests(modelPath, testFile, orgFile, consts, saveLabels = False):
    model, _ = LoadModel(modelPath, consts) 
    modelName = os.path.splitext(os.path.basename(modelPath))[0]   
    # PrintToLog(f"Batch size: {consts.BatchSize()}")
    PrintToLog(f"Model loaded: {modelName}")
    PrintToLog(f"Test file: {testFile}")
    PrintToLog("---------------------------------------------------------------")

    if(not (orgFile is None)):
        true_points_labels = ReadXYZL(orgFile)
        true_points = true_points_labels[:,:3]
        true_labels = true_points_labels[:,3]
        
        # label_map = {
        #     1 : 0, # unclassified -> unlabeled points
        #     2 : 1, # ground -> "natural terrain"
        #     5 : 3, # high veg -> high veg
        #     3 : 4, # low veg -> low veg        
        #     6 : 5, # building -> building
        #     19 : 6, # hard scape
        #     7 : 7, # low point -> artefacts
        #     20 : 8, # cars
            
        #     15 : 6, # poles and signs -> hard scape
        #     16 : 6, # wires -> hard scape     
        # }
        # true_labels = ChangeLabels(true_labels, label_map)

        true_points, _, true_labels = consts.NormalizeData(true_points, labels = true_labels, validation = True)        
    else:
        true_points = None
        true_labels = None    

    # MeasureInference(model, testFile, consts, 6, true_points, true_labels, saveLabels, modelName=modelName)
    # MeasureInference(model, testFile, consts, 4, true_points, true_labels, saveLabels, modelName=modelName)
    # MeasureInference(model, testFile, consts, 12, true_points, true_labels, saveLabels, modelName=modelName)
    # MeasureInference(model, testFile, consts, 8, true_points, true_labels, saveLabels, modelName=modelName)
    MeasureInference(model, testFile, consts, 4, true_points, true_labels, saveLabels, modelName=modelName)
    # MeasureInference(model, testFile, consts, 5, true_points, true_labels, saveLabels, modelName=modelName)
    # MeasureInference(model, testFile, consts, 4, true_points, true_labels, saveLabels, modelName=modelName)
    # MeasureInference(model, testFile, consts, 3, true_points, true_labels, saveLabels, modelName=modelName)
    # MeasureInference(model, testFile, consts, 2, true_points, true_labels, saveLabels, modelName=modelName)
    # MeasureInference(model, testFile, consts, 1, true_points, true_labels, saveLabels, modelName=modelName)
    # MeasureInference(model, testFile, consts, 0.5, true_points, true_labels, saveLabels, modelName=modelName)

def VisualizeModelConfidence():
    model, _ = LoadModel(r"C:\Users\jstanke\Downloads\Sem3DExt(12-4-4)(noFeature)(Scale)(Rotate)_0990e1f922_35_train(81.2)_val(64.7).h5", consts)    
    seq = TrainSequence(10, consts, randomPoints = True, validation = True)
    for i in range(10):
        data = seq.__getitem__(i)
        pts = data[0][0]
        true_lbl = np.argmax(data[1], axis=-1)
        
        pred = model.predict(pts)
        pred_lbl = np.argmax(pred, axis=-1)
        
        for y in range(len(true_lbl)):          
            confidenceColor = np.tile([[0,0,1.]], [len(pts[y]), 1])
            
            mean_confidence = [[i, np.mean(pred[y][np.where(true_lbl[y] == i)[0]]) ] for i in range(data[1].shape[-1])]
            
            confidence = np.expand_dims(pred[y].max(-1), axis=-1)
            confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min())
            confidenceColor += np.tile([[1,0,-1]], [len(pts[y]), 1]) * confidence
            DataTool().VisualizePointCloudAsync([pts[y]], dataColors=[confidenceColor], windowName="Confidence")
            
            DataTool().VisualizePointCloudAsync([pts[y]], classes = [true_lbl[y]], windowName="Ground true")
            DataTool().VisualizePointCloudAsync([pts[y]], classes = [pred_lbl[y]], windowName="Prediction")
            
            print(data[0][0])

if __name__ == "__main__":    
    modelPath = None
        
    # consts = NPM3D()
    # consts = SDE()
    consts = Sem3DExtended()
    # consts = NMG()
    
    # consts.classCount = SDE.classCount
    # consts.class_color = SDE.class_color
    # consts.noFeature = SDE.noFeature
    # consts.classNames = SDE.classNames
        
    # Semantic3D.noFeature = consts.noFeature
    # consts.ConcatDataForTraining(Semantic3D)
    # consts = LaneMarkingsLargeSparse()
    # consts = LaneMarkingsSmallDense()
    # consts = Curbs()
    # consts = Aerial()

    consts.noFeature = True
    # consts.Fusion = True
    consts.Scale = True
    consts.Rotate = True
    # consts.Mirror = True
    # consts.Jitter = True
    # consts.FtrAugment = True
    
    # seq = TrainSequence(100, consts, randomPoints = True, balanceClasses = consts.classBalanceTrain, validation = False)
    # for i in range(len(seq)):
    #     data = seq.__getitem__(i)
    #     for y in range(len(data[0][0])):
    #         DataTool().VisualizePointCloud([data[0][0][y]], classes = [data[1][y]])

    setConfig = False
    # if(not Config.IsLocalMachine()):
    try:
        wandb.init(project='ConvPointSemanticSegmentation', entity='vuteam')
    except:
        print("wandb not configured.")
            
    config = wandb.config        
    config.batchSize = consts.BatchSize()
    config.npoints = consts.npoints
    config.featureComponents = consts.featureComponents
    config.classCount = consts.classCount
    config.blocksize = consts.blocksize
    config.input_tile_count = consts.input_tile_count
    config.test_step = consts.test_step
    config.name = consts.Name()
    config.trainFiles = np.sum([len(consts.TrainFiles()) for cts in consts.datasets]) if len(consts.datasets) > 0 else len(consts.TrainFiles())
    config.trainFileNames = list(np.concatenate([cts.trainFiles for cts in consts.datasets])) if len(consts.datasets) > 0 else consts.trainFiles
    config.validationFiles = np.sum([len(cts.ValidationFiles()) for cts in consts.datasets]) if len(consts.datasets) > 0 else len(consts.ValidationFiles())
    config.validationFileNames = list(np.concatenate([cts.validationFiles for cts in consts.datasets])) if len(consts.datasets) > 0 else consts.validationFiles
    config.testFiles = np.sum([len(cts.TestFiles()) for cts in consts.datasets]) if len(consts.datasets) > 0 else len(consts.TestFiles())
    config.testFileNames = list(np.concatenate([cts.testFiles for cts in consts.datasets])) if len(consts.datasets) > 0 else consts.testFiles
    config.TrainFileBias = str(consts.trainFileBias).split('.')[1]
    config.ValFileBias = str(consts.valFileBias).split('.')[1]
    
    if(len(consts.datasets) > 0):
        config.trainFilesBySet = [[cts.name, len(cts.TrainFiles())] for cts in consts.datasets]
        config.validationFilesBySet = [[cts.name, len(cts.ValidationFiles())] for cts in consts.datasets]    
    
    config.testFiles = len(consts.TestFiles())
    
    config.classBalanceTrain = consts.classBalanceTrain
    config.classBalanceTest = consts.classBalanceTest

    config.noFeature = consts.noFeature
    config.Fusion = consts.Fusion
    config.Scale = consts.Scale
    config.Rotate = consts.Rotate
    config.Mirror = consts.Mirror
    config.Jitter = consts.Jitter
    config.FtrAugment = consts.FtrAugment
    config.class_weights = consts.class_weights
    config.testsAreFixed = consts.testsAreFixed
    config.datasets = [cts.name for cts in consts.datasets]
    config.TestInterval = consts.TestInterval
    config.ValidationInterval = consts.ValidationInterval
    
    PrintToLog(str(config))

    # modelPath = "Aerial(676&168)(Rotate)_3dc13635745344d8aa77f8d23ef9eb74_9_train(88.7)_val(82.2)"
    # modelPath = "Curbs(7&1)(noFeature)(Rotate)_21bdbe6aa82d4e259526ab46577e795a_25_train(75.1)_val(60.7).h5"
    # modelPath = "Sem3D(vox)(RGB)(FullAugment)_55_train(85.7)_val(79.9)"
    # modelPath = ["Sem3D(vox)(RGB)(FullAugment)_55_train(85.7)_val(79.9)", "Sem3D(NOCOL)_50_train(87.4)_val(69.1)"]
    # modelPath = ["NPM3D(80&5)(RGB)(NoScale)_28_train(88.3)_val(73.2).h5", "NPM3D(80&5)(NOCOL)(FullAugment)_28_train(87.3)_val(71.5).h5"]
    # modelPath = LatestModel("Sem3D(14&1)(noFeature)(Scale)(Rotate)(Mirror)(Jitter)")
    # modelPath = LatestModel(consts.Name())    

    # model = CreateModel(consts.classCount, 1, noColor=True)
    # samples = 10
    # pts = np.random.uniform(0, 100, (samples, consts.npoints, Const.pointComponents))   
    # lbl = np.random.randint(0, consts.classCount, (samples, consts.npoints))
    # lbl = np.eye(consts.classCount)[lbl]
    # model.fit(pts, lbl, batch_size=1, epochs = 10)

    # if(isinstance(modelPath,list)):
    #     consts.Fusion = True

    # if(not consts.Fusion and not consts.IsLocalMachine()):
    #     tf.config.optimizer.set_jit(True) #Gives more than 10% boost!!!
    #     print("XLA enabled.")

    # modelPath = ["Sem3D(14&1)(noFeature)(Scale)(Rotate)(Mirror)(Jitter)_9bbee708a7814063af9d85070452abd8_59_train(85.2)_val(72.8)", 
    #             "Sem3D(14&1)(noFeature)(Rotate)(Mirror)(Jitter)_ff2eb229084247d9a1c63caa519e9890_58_train(84.9)_val(75.5)",
    #             "Sem3D(14&1)(noFeature)_dffc17f77e924894bbdbdad818ab6994_40_train(85.1)_val(68.8)"]
    # EvaluateModels([modelPath], testFiles, consts)

    # seq = TrainSequence(10, consts.datasets[1], randomPoints = True, balanceClasses = config.classBalanceTrain, validation = True, mapLabels="SDE")
    # for i in range(10):
    #     data = seq.__getitem__(i)
    #     print(data[0][0])    
    
    # consts = SDE()
    # consts.noFeature = True
    
    # modelPath = "C:/Users/jstanke/Downloads/SDE(20-6-0)(noFeature)(Rotate)_d8c2247b66_42_train(78.6)_val(67.8).h5"
    # model, _ = LoadModel(modelPath, consts)
    # voxelized = "E:/ParisData/preprocessedTrain/Athis.las"
    # original = "E:/ParisData/eCog command-line for inference/input/Athis.las"
    # output = "E:/ParisData/eCog command-line for inference/output/Athis_4cls.las"
    # GenerateLargeFile(model, voxelized, original, consts, output, Upscale = True)

    # TrainModel(trainFiles, testFiles, consts, modelPath = modelPath)# , epochs = 8) #continue train
    TrainModel(consts, trainSteps = 2000, testSteps = 400) #new model

    # modelPath = HighestValMIOUModel("NPM3D(80&5)(fusion)(FullAugment)")

    #NPM3D
    # GenerateData(modelPath, Paths.GetFiles(consts.Paths.rawTest), consts, consts.Paths.generatedTest)
    
    # file = "skate_park_clean"
    # src_pts = ReadXYZ(f"/media/jonas/SSD Extreme/semantic3d/processedTest/{file}.npy")
    # src_lbl = ReadLabels(f"/media/jonas/SSD Extreme/semantic3d/processedTest/{file}_step_2_voxels_lbl.npy")
    # org_pts = ReadXYZ(f"/media/jonas/SSD Extreme/semantic3d/rawTest/{file}.las")
    # pred_lbl = f"/media/jonas/SSD Extreme/semantic3d/rawTest/{file}_predlbl.npy"
    # UpscaleToOriginal(org_pts, src_pts, src_lbl, pred_lbl)

    # file = "MM_Boulder"
    # pts = ReadXYZ(f"/media/jonas/SSD Extreme/semantic3d/processedTest/{file}.npy")  
    # lbl = ReadLabels(f"/media/jonas/SSD Extreme/semantic3d/processedTest/{file}_step_2_voxels_lbl.npy")
    # DataTool().VisualizePointCloudClasses([pts], [lbl])

    #Semantic3D    
    # raw_file = None
    # modelPath = "Sem3D(vox)(RGB)(FullAugment)_55_train(85.7)_val(79.9)" ; consts.noFeature = False
    # modelPath = "Sem3D(13&2)(noFeature)(Scale)(Rotate)(Mirror)(Jitter)_ea03acfa13334ee29d53c92e2094e34d_3_train(35.8)_val(33.0)" ; consts.noFeature = True
    # voxel_file = consts.Paths.processedTrain+"domfountain_station1_xyz_intensity_rgb_voxels.npy"
    # raw_file = consts.Paths.rawTrain+"domfountain_station1_xyz_intensity_rgb.hdf5"
    # voxel_file = consts.Paths.processedTest+"MM_SmallArea.npy"
    # voxel_file = consts.Paths.processedTest+"skate_park.npy"
    # voxel_file = consts.Paths.processedTest+"MM_Boulder.npy"
    # raw_file = consts.Paths.rawTest+"MM_Boulder.las"    
    # SlidingOverlapTests(modelPath, voxel_file, raw_file, consts)


    # SlidingOverlapTests(modelPath, consts.Paths.processedTest+"MM_Boulder.npy", None, consts, saveLabels = True)
    # SlidingOverlapTests(modelPath, consts.Paths.processedTest+"skate_park.npy", None, consts, saveLabels = True)
    # SlidingOverlapTests(modelPath, "/media/jonas/SSD Extreme/semantic3d/processedTest2/"+"skate_park_clean.npy", None, consts, saveLabels = True)
    # SlidingOverlapTests(modelPath, "/media/jonas/SSD Extreme/semantic3d/processedTest/"+"skate_park_clean.npy", None, consts, saveLabels = True)
    # SlidingOverlapTests(modelPath, consts.Paths.processedTest+"MM_SmallArea.npy", None, consts, saveLabels = True)
    # SlidingOverlapTests(modelPath, consts.Paths.processedTest+"MM_SmallArea_clean.npy", None, consts, saveLabels = True)
    
    
    # GenerateLargeData(modelPath, voxelized, original, consts, output, Upscale=True)    
    # UpscaleFilesAsync(modelPath, Paths.Semantic3D.processedTest, Paths.Semantic3D.rawTest, Paths.Semantic3D.generatedTest)
    # RenameSemantic3DFiles(Paths.Semantic3D.generatedTest + Paths.FileName(modelPath))

    #Curbs
    # EvaluateModels([modelPath], testFiles, consts)
    # GenerateData(modelPath, testFiles, consts, consts.Paths.pointCloudPath+"/generated/")
    # GenerateLargeData(modelPath, testFiles, consts, consts.Paths.pointCloudPath+"/generated/")