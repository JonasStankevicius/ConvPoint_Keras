from imports import *
import threading
import pandas as pd
from plyfile import PlyData
from sklearn.decomposition import PCA
from multiprocessing import Process
import random
from tqdm import tqdm
import laspy
from functools import partial

from sklearn.neighbors import KDTree

if(os.path.exists("C:/") or os.path.exists("/home/jonas")):
    import open3d as o3d    

def SaveRenderOptions(vis):
    print("Saving camera parameters")
    
    params = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters("./data/camera.json", params)

    return False

def MaskTrajectoryFile(vis):
    trajectory_file = "./data/camera_trajectory.json"
    mask_trajectory_file = "./data/camera_trajectory.json.mask"
    
    if(os.path.exists(trajectory_file)):
        os.rename(trajectory_file, mask_trajectory_file)
    elif(os.path.exists(mask_trajectory_file)):
        os.rename(mask_trajectory_file, trajectory_file)

    return False

class TrajectoryRecorder():    
    def __init__(self):
        self.trajectory = []
        self.trajectory_file = "./data/camera_trajectory.json"

        if(os.path.exists(self.trajectory_file)):
            os.remove(self.trajectory_file)

    def record(self, vis):
        params = vis.get_view_control().convert_to_pinhole_camera_parameters()
        self.trajectory.append(params)

    def save(self, vis):
        trajectory = o3d.camera.PinholeCameraTrajectory()
        trajectory.parameters = self.trajectory
        o3d.io.write_pinhole_camera_trajectory(self.trajectory_file, trajectory)

    def delete(self, vis):
        self.trajectory = []
        
        if(os.path.exists(self.trajectory_file)):
            os.remove(self.trajectory_file)

def AppendCameraTrajectory(vis):
    print("Append camera trajectory.")

    params = vis.get_view_control().convert_to_pinhole_camera_parameters()
    trajectory = o3d.camera.PinholeCameraTrajectory()

    if(os.path.exists("./data/camera_trajectory.json")):
        trajectory = o3d.io.read_pinhole_camera_trajectory("./data/camera_trajectory.json")
    
    trajectory.parameters = trajectory.parameters + [params]
    
    o3d.io.write_pinhole_camera_trajectory("./data/camera_trajectory.json", trajectory)

    return False

def LoadRenderOptions(vis, returnVis = False):
    # time.sleep(1) # sleep 1 second

    paramsFile = "./data/camera.json"
    if(not os.path.exists(paramsFile)):
        return False

    print("Loading camera parameters")
    params = o3d.io.read_pinhole_camera_parameters(paramsFile)
    vis.get_view_control().convert_from_pinhole_camera_parameters(params)

    if(returnVis):
        return vis
    else:
        return False

def AnimationCallBack(vis):
    ctr = vis.get_view_control()
    ctr.rotate(0.2, 0.0)
    # ctr.scale(1/80)
    return False

class PlayTrajectory():
    def __init__(self):
        assert(os.path.exists("./data/camera_trajectory.json"))
        self.trajectory = o3d.io.read_pinhole_camera_trajectory("./data/camera_trajectory.json").parameters
        self.i = 0
        self.time = time()
    
    def StepTrajectory(self, vis):
        if(self.i < len(self.trajectory)): # and time() - self.time > 1):
            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(self.trajectory[self.i])
            
            self.time = time()
            self.i += 1

class DataTool:    
    def __init__(self, piece_size = 1000000, threads_allowed = 1000):
        self.piece_size = piece_size
        self.threads_allowed = threads_allowed

        self.vis = None
        self.bBox = None
        self.pointCloud = None
        self.displayCloud = None
    
    def ReadPointCloudTxt(self, path, pcData):            
        t = time() 

        pcFile = open(path, 'r')
        self.pointCloud = None

        threads = [None] * self.threads_allowed    
        points_read = 0
        thread_index = 0
        
        while True:
            if(threads[thread_index] is not None and threads[thread_index].isAlive()):
                print("Wait for thread {}".format(thread_index), end="                  \r")
                threads[thread_index].join()
        
            chunk = pcFile.readlines(self.piece_size)            

            if(len(chunk) < 1):
                break
            
            if(pcData.shape[0] <= points_read + len(chunk)):
                if(pcData.shape[0] == 0):
                    pcData.resize((points_read + len(chunk))*2, axis=0)
                else:
                    pcData.resize(pcData.shape[0]*2, axis=0)

            # if(type(self.pointCloud) is np.ndarray):                
                # self.pointCloud = np.append(self.pointCloud, np.zeros(shape=(len(chunk), 7), dtype="float32"), axis=0)
            # else:
                # self.pointCloud = np.zeros(shape=(len(chunk), 7), dtype="float32")

            threads[thread_index] = threading.Thread(target= self.__ReadPCChunkTxt, args=(chunk, points_read, pcData))
            threads[thread_index].start()

            points_read += len(chunk)

            thread_index += 1
            if(thread_index >= self.threads_allowed):
                thread_index = 0

            print("{0} points read".format(points_read), end='\r')
        
        for i in range(self.threads_allowed):
            if(threads[i] is not None):
                print("Join thread {}".format(i), end="                  \r")
                threads[i].join()
        
        pcData.resize(points_read, axis=0)
        pcFile.close()
        print("PC Finished reading {} points in {:.2f} min".format(pcData.shape[0], (time() - t)/60))
        return self.pointCloud
    
    def __ReadPCChunkTxt(self, chunk, start_index, pcData):        
        for i in range(len(chunk)):
            if(chunk[i] != ""):
                flts = chunk[i].replace('\n','').split()
                # self.pointCloud[start_index + i] = np.array([float(flts[0]), float(flts[1]), float(flts[2]), float(flts[3]),
                #                                             float(flts[4]), float(flts[5]), float(flts[6])])
                pcData[start_index + i] = np.array([float(flts[0]), float(flts[1]), float(flts[2]), float(flts[3]),
                                                            float(flts[4]), float(flts[5]), float(flts[6])])
        del chunk

    def ReadPointLabelsTxt(self, path):
        t = time()        

        labelsFile = open(path, 'r')
        labelsArr = labelsFile.read().split('\n')
        
        if(labelsArr[-1] == ''):
            del labelsArr[-1]

        self.labels = np.array(labelsArr, dtype='int')

        print("Finished reading {} labels in {:.2f} min".format(self.labels.shape[0], (time() - t)/60))
        return self.labels
        
    def ConvertToBin(self, path_to_pointcloud, path_to_pointlabels, output_path, extension = ".hdf5"):
        if(os.path.isfile(output_path+extension)):
            return
        else:
            print("Converting: ",output_path)

        t = time()
        pointcloud = np.array(pd.read_csv(path_to_pointcloud, sep=" ", dtype=np.float32, header=None), dtype=np.float32)
        
        h5File = None
        if(extension == ".hdf5"):
            h5File = h5py.File(output_path+".hdf5", 'w')
            h5File.create_dataset("pointcloud", data=pointcloud, dtype='float32', compression="lzf")            
            del pointcloud
        
        if(path_to_pointlabels):
            labels = np.array(pd.read_csv(path_to_pointlabels, dtype=np.int8, header=None))
            
            if(extension == ".hdf5"):
                h5File.create_dataset("labels", data=labels, dtype='int8', compression="lzf")                
            elif(extension == ".npy"):
                pointcloud = np.concatenate((pointcloud, labels.astype(np.float32)), 1)            
            del labels
        
        print("Done reading")

        if(extension == ".hdf5"):
            h5File.close()
        elif(extension == ".npy"):
            np.save(output_path, pointcloud, allow_pickle=False)
        
        print("done in {}:{} min.".format(int((time() - t)/60), int((time() - t)%60)))
    
    def ConvertDatasets(self, folder, outputFolder):        
        pcFiles = [f for f in listdir(folder) if isfile(join(folder, f)) and f.endswith('.txt')]
        
        os.makedirs(outputFolder, exist_ok=True)
        for file in pcFiles:
            name = file.replace('.txt', '')

            if(not isfile(join(folder, name+'.labels'))):
                self.ConvertToBin(join(folder, name+'.txt'), None, join(outputFolder, name))
            else:
                self.ConvertToBin(join(folder, name+'.txt'), join(folder, name+'.labels'), join(outputFolder, name))

    def createWindow(self, windowName = "Pointcloud"):
        # vis = o3d.visualization.Visualizer()        
        self.vis = o3d.visualization.VisualizerWithKeyCallback()

        self.vis.create_window(windowName, 800, 800)

        opt = self.vis.get_render_option()
        # opt.background_color = np.asarray([0, 0, 0])    

    def addPointCloud(self, pointCloud, downSample = False, color = None):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pointCloud)
        
        if(color != None):
            pc.paint_uniform_color(np.asarray(color))

        if(downSample):
            pc = o3d.geometry.voxel_down_sample(pc, voxel_size=0.02)

        self.vis.add_geometry(pc)

    def setPointCloud(self, pointCloud, downSample = False, color = None):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pointCloud)

        if(downSample):
            pc = o3d.geometry.voxel_down_sample(pc, voxel_size=0.02)
        
        if(self.pointCloud == None):
            self.pointCloud = pc
        else:
            self.pointCloud.points = pc.points
        
        if(color != None):
            self.pointCloud.paint_uniform_color(np.asarray(color))

    def addBoundingBox(self, bBox, color = []):
        self.addBbox(self.vis, bBox, color)
    
    @staticmethod
    def addBbox(vis, bBox, color = []):
        lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],[0, 4], [1, 5], [2, 6], [3, 7]]

        box =  [[bBox[0], bBox[2], bBox[4]], 
                [bBox[1], bBox[2], bBox[4]], 
                [bBox[0], bBox[3], bBox[4]], 
                [bBox[1], bBox[3], bBox[4]],
                [bBox[0], bBox[2], bBox[5]], 
                [bBox[1], bBox[2], bBox[5]], 
                [bBox[0], bBox[3], bBox[5]], 
                [bBox[1], bBox[3], bBox[5]]]
            
        if(len(color) == 0):
            colors = [[1,0,0] for _ in range(len(lines))]
        else:
            colors = [color for _ in range(len(lines))]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.array(box))
        line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
        line_set.colors = o3d.utility.Vector3dVector(np.array(colors))

        vis.add_geometry(line_set)

    def setBoundingBox(self, bBox, color = None):        
        box =  [[bBox[0], bBox[2], bBox[4]], 
                [bBox[1], bBox[2], bBox[4]], 
                [bBox[0], bBox[3], bBox[4]], 
                [bBox[1], bBox[3], bBox[4]],
                [bBox[0], bBox[2], bBox[5]], 
                [bBox[1], bBox[2], bBox[5]], 
                [bBox[0], bBox[3], bBox[5]], 
                [bBox[1], bBox[3], bBox[5]]]
            
        if(color == None):
            colors = [[1,0,0] for _ in range(12)] #len(lines)
        else:
            colors = [color for _ in range(12)]
    
        if(self.bBox == None):
            lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],[0, 4], [1, 5], [2, 6], [3, 7]]

            line_set = o3d.geometry.LineSet()
            line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
            line_set.points = o3d.utility.Vector3dVector(np.array(box))            
            line_set.colors = o3d.utility.Vector3dVector(np.array(colors))
            self.bBox = line_set
        else:
            self.bBox.points = o3d.utility.Vector3dVector(np.array(box))
            self.bBox.colors = o3d.utility.Vector3dVector(np.array(colors))

    def setDisplayedCloud(self, bBox):
        if(self.pointCloud == None or bBox == None):
            return

        points = np.asarray(self.pointCloud.points)

        rows = np.where((points[:,0] >= bBox[0]) & 
                        (points[:,0] <= bBox[1]) &

                        (points[:,1] >= bBox[2]) & 
                        (points[:,1] <= bBox[3]) &

                        (points[:,2] >= bBox[4]) & 
                        (points[:,2] <= bBox[5]) )
        
        if(self.displayCloud == None):
            self.displayCloud = o3d.geometry.PointCloud()       

        self.displayCloud.points = o3d.utility.Vector3dVector(points[rows])

    def VisualizePointCloudAsync(self, dataset, dataColors = None, classes = None, downSample = False, deleteZeros = False, bBoxes = None, boxesColors = [], windowName = None, animationFunction = None, loadCameraSettings = False, recordTrajectory = False):
        p = Process(target=self.VisualizePointCloud, args=(dataset, dataColors, classes, downSample, deleteZeros, bBoxes, boxesColors, windowName, animationFunction, loadCameraSettings, recordTrajectory))
        p.start()

    def GetMainColors(self):
        grey = np.array([90, 90, 90])/255
        red = np.array([136, 0, 1])/255
        mint = np.array([170, 255, 195])/255
        teal = np.array([0, 128, 128])/255
        green = np.array([60, 180, 75])/255
        verygreen = np.array([0, 255, 0])/255
        brown = np.array([170, 110, 40])/255
        black = np.array([0, 0, 0])/255
        blue = np.array([0, 0, 255])/255    
        pink = np.array([255, 56, 152])/255
        return [grey, red, mint, teal, blue, verygreen, brown, green]

    def VisualizePointCloudClasses(self, dataset, labels, downSample = False, bBoxes = None, boxesColors = [], windowName = "Pointcloud", animationFunction = None, loadCameraSettings = False, recordTrajectory = False):
        import matplotlib.colors as mcolors

        assert(len(dataset) == len(labels))

        pointSets = []
        colors = []

        for i in range(len(dataset)):
            assert(len(dataset[i]) == len(labels[i]))

            classes = np.unique(labels[i])
            # colors.extend([mcolors.to_rgb(col) for col in np.random.choice(list(mcolors.CSS4_COLORS), size=len(classes))])
            colors.extend(self.GetMainColors()[:len(classes)])

            for cl in classes:
                pointSets.append(dataset[i][np.where(labels[i] == cl)[0]])

        self.VisualizePointCloudAsync(pointSets, dataColors = colors, downSample = downSample, bBoxes = bBoxes, boxesColors=boxesColors, windowName=windowName, animationFunction=animationFunction, loadCameraSettings=loadCameraSettings, recordTrajectory=recordTrajectory)

    def VisualizePointCloud(self, dataset, dataColors = None, classes = None, downSample = False, deleteZeros = False, bBoxes = None, boxesColors = [], windowName = None, animationFunction = None, loadCameraSettings = False, recordTrajectory = False):
        if(len(dataset) == 0):
            return
        if(isinstance(dataset, np.ndarray)):
            dataset = [dataset]
        if(isinstance(dataColors, np.ndarray) or (isinstance(dataColors, list) and not isinstance(dataColors[0], list) and not isinstance(dataColors[0], np.ndarray))):
            dataColors = [dataColors]        
        
        # windowName = "".join([ f"[Class {i} : {classCount[i]} points]" for i in range(len(classCount)) if classCount[i] > 0])
        if(windowName is None):
            windowName = f"Point count: {np.sum([len(data) for data in dataset])}"
        self.createWindow(windowName=windowName)

        for i in range(len(dataset)):
            if(dataset[i] is None):
                continue
            if(len(dataset[i]) == 0):
                continue
            
            dataset[i] = np.array(dataset[i])
            if (deleteZeros):
                if(len(dataset[i][0]) == 3):
                    indexes = np.where((dataset[i][:, 0] == 0.0) & (dataset[i][:, 1] == 0.0) & (dataset[i][:, 2] == 0.0))
                else:
                    indexes = np.where((dataset[i][:, 0] == 0.0) & (dataset[i][:, 1] == 0.0) & (dataset[i][:, 2] == 0.0) & (dataset[i][:, 3] == 0.0))
                dataset[i] = np.delete(dataset[i], indexes, axis=0)
            
            print("Adding dataset {}/{} to visualization    ".format(i+1, len(dataset)), end = '\r')
            pc = o3d.geometry.PointCloud()
            if(len(dataset[i][0]) == 3):                
                pc.points = o3d.utility.Vector3dVector(dataset[i])          
            else:
                pc.points = o3d.utility.Vector3dVector(dataset[i][:,:3])
            
            if(not (dataColors is None)):
                if(not (dataColors[i] is None)):
                    if(len(dataColors[i]) == len(dataset[i]) and len(dataset[i]) != 3):
                        # handle grey scale intensity
                        if(dataColors[i].shape[-1] == 1):
                            dataColors[i] = np.tile(dataColors[i], [1, 3])
                        pc.colors = o3d.utility.Vector3dVector(np.asarray(dataColors[i]))
                    elif(len(dataColors) == len(dataset)):
                        pc.paint_uniform_color(np.asarray(dataColors[i]))
            if(not (classes is None)):
                classCount = np.zeros((1000), dtype=int)
                if(not (classes[i] is None)):
                    assert len(classes[i]) == len(dataset[i])
                    assert len(classes[i].shape) <= 2
                    
                    if(len(classes[i].shape) == 2 and classes[i].shape[1] != 1):
                        classes[i] = np.argmax(classes[i], axis = 1)
                    if(len(classes[i].shape) > 1):
                        classes[i] = classes[i].reshape(-1)
                        
                    colors = np.zeros((len(classes[i]), 3), np.float32)
                    for lbl in np.unique(classes[i]):                        
                        lbl = int(lbl)
                        idx = np.where(classes[i] == lbl)[0]
                        classCount[lbl] += len(idx)
                        colors[idx] = np.random.uniform(0, 1, 3)
                    pc.colors = o3d.utility.Vector3dVector(colors)                    
                                
                        
            if(downSample):
                pc = o3d.geometry.PointCloud.voxel_down_sample(pc, voxel_size=0.02)                    

            self.vis.add_geometry(pc)

        print("")
        
        if(bBoxes is not None):
            print("Adding {} bBoxes to visualization".format(len(bBoxes)), end = '\r')
            for i in range(len(bBoxes)):
                # print("Adding bBox {}/{} to visualization".format(i+1, len(bBoxes)), end = '\r')

                color = []
                if(len(boxesColors) > i and boxesColors[i] is not None):
                    color = boxesColors[i]

                self.addBoundingBox(bBoxes[i], color)

        self.vis.register_key_callback(ord("s"), SaveRenderOptions)
        self.vis.register_key_callback(ord("S"), SaveRenderOptions)

        self.vis.register_key_callback(ord("l"), LoadRenderOptions)
        self.vis.register_key_callback(ord("L"), LoadRenderOptions)

        self.vis.register_key_callback(ord("m"), MaskTrajectoryFile)
        self.vis.register_key_callback(ord("M"), MaskTrajectoryFile)

        if recordTrajectory:
            recorder = TrajectoryRecorder()

            self.vis.register_key_callback(ord("a"), recorder.record)
            self.vis.register_key_callback(ord("A"), recorder.record)

            self.vis.register_key_callback(ord("r"), recorder.save)
            self.vis.register_key_callback(ord("R"), recorder.save)

            self.vis.register_key_callback(ord("d"), recorder.delete)
            self.vis.register_key_callback(ord("D"), recorder.delete)

        # paramFiles = "./data/camera.json"
        # if(os.path.exists(paramFiles)):
        #     os.remove(paramFiles)

        if not (animationFunction is None):
            self.vis.register_animation_callback(animationFunction)

        if(loadCameraSettings):
            self.vis = LoadRenderOptions(self.vis, returnVis=True)

        self.vis.run()
        self.vis.destroy_window()     

    def DoBoxesQA(self, pointcloud = None, bBoxes = None, downSamplePC = False):
        if(len(pointcloud) == 0 and len(self.pointCloud) == 0):
            return
        elif(len(pointcloud) != 0):
            self.setPointCloud(pointcloud, downSamplePC)

        acceptedBoxes = []
        
        def darkMode(vis):
            opt = vis.get_render_option()
            opt.background_color = np.asarray([0, 0, 0])
            return False

        def acceptBox(vis):
            print("Accept")
            acceptedBoxes.append(bBoxes[self.boxIndex])
            vis.close()            
            return False
        
        def discardBox(vis):
            print("Discard")
            vis.close()            
            return False

        key_to_callback = {}
        key_to_callback[ord("Y")] = acceptBox
        key_to_callback[ord("N")] = discardBox
        key_to_callback[ord("D")] = darkMode

        self.boxIndex = 0
        for box in bBoxes:
            self.setDisplayedCloud(box)
            self.setBoundingBox(box)
            o3d.visualization.draw_geometries_with_key_callbacks([self.displayCloud, self.bBox], key_to_callback, "QA", 800, 800)
            self.boxIndex += 1

        print("QA done")
        return acceptedBoxes
    
    def QAResults(self, dataFolder, boxesFolder, boxesExportFolder, override = True):
        pcFiles = [f for f in listdir(boxesFolder) if isfile(join(boxesFolder, f)) and f.endswith('.txt')]
        
        for file in pcFiles:
            name = file.replace('.txt', '').replace('BBOXES_', '')

            boxesFile = join(boxesFolder, 'BBOXES_'+name+'.txt')
            dataFile = join(dataFolder, name+'.hdf5')
            newBoxPath = join(boxesExportFolder, 'BBOXES_'+name+'.txt')

            if(isfile(dataFile)):
                if(override == False and isfile(newBoxPath)):
                    print("Already done: "+dataFile)
                    continue

                print("QA: "+dataFile)

                pc = self.ReadHDF5XYZ(dataFile)
                boxes = ReadBoundingBoxes(boxesFile)
                newBoxes = self.DoBoxesQA(pc, boxes, True)
                
                SaveBoundingBoxes(newBoxPath, newBoxes)

def ReadHDF5(path, with_labels = True):
    print("Reading '{}'".format(path))

    t=time()

    h5File = h5py.File(path, 'r')
    pointCloud = np.array(h5File["pointcloud"], dtype="float32")

    if(with_labels):
        labels = np.array(h5File["labels"], dtype="float32")
        labels = np.expand_dims(labels, 1)
        pointCloud = np.append(pointCloud, labels, axis=1)
        del labels

    print("Finished reading in {:.2f} min. Shape = {}".format((time() - t)/60, pointCloud.shape))

    return pointCloud
    
def ReadXYZ(file, dataName = "pointcloud", verbose = False, readFormat=None):
    if(verbose):
        print("Reading pointcloud of '{}'".format(path))

    t=time()

    xyz = None

    if(file.endswith(".hdf5")):
        h5File = h5py.File(file, 'r')
        pc = h5File["pointcloud"]
        xyz = pc[:, :3]
        h5File.close()
    if(file.endswith(".npy") or readFormat == ".npy"):
        pc = np.load(file)
        xyz = pc[:, :3]
    elif(file.endswith(".las") or file.endswith(".laz")):
        with laspy.open(file) as lasFile:
            lasFile = lasFile.read()
        xyz = np.concatenate((np.expand_dims(lasFile.x,1), np.expand_dims(lasFile.y,1), np.expand_dims(lasFile.z,1)), 1)
        # xyz *= lasFile.header.scales
        xyz -= np.array([lasFile.header.x_min, lasFile.header.y_min, lasFile.header.z_min])
    elif(file.endswith(".ply")):
        plydata = PlyData.read(file)
        x = plydata["vertex"].data["x"].astype(np.float32)
        y = plydata["vertex"].data["y"].astype(np.float32)
        z = plydata["vertex"].data["z"].astype(np.float32)
        xyz = np.concatenate((np.expand_dims(x,1), np.expand_dims(y,1), np.expand_dims(z,1)), axis=1)
        
    print("Finished reading pointcloud in {:.2f} min. Shape = {}".format((time() - t)/60, xyz.shape))

    return xyz
    
def ReadRGB(file, dataName = "pointcloud", verbose = False):
    t=time()

    rgb = None

    if(file.endswith(".hdf5")):
        h5File = h5py.File(file, 'r')
        pc = h5File["pointcloud"]
        rgb = pc[:, 4:7]
        h5File.close()
    if(file.endswith(".npy")):
        pts = np.load(file)
        rgb = pts[:, 3:6]
    elif(file.endswith(".las")):
        with laspy.open(file) as lasFile:
            lasFile = lasFile.read()
        rgb = np.concatenate((np.expand_dims(lasFile.Red,1), np.expand_dims(lasFile.Green,1), np.expand_dims(lasFile.Blue,1)), 1)
        rgb = rgb/65536 #[0,1]
        lasFile.close()
    elif(file.endswith(".ply")):
        plydata = PlyData.read(file)
        fts = plydata["vertex"].data["reflectance"].astype(np.float32)
        fts = np.expand_dims(fts, 1)
    
    print("Finished reading RGB values in {:.2f} min. Shape = {}".format((time() - t)/60, rgb.shape))

    return rgb

def ReadXYZRGBLBL(file, dataName = "pointcloud", verbose = False):
    data = None
    if(file.endswith(".las") or file.endswith(".laz")):
        lasFile = None
        with laspy.open(file) as lasFile:
            lasFile = lasFile.read()
        
        xyz = np.concatenate(
            (np.expand_dims(lasFile.x, 1), np.expand_dims(lasFile.y, 1), np.expand_dims(lasFile.z, 1)), 1
        )

        intensity = np.expand_dims(lasFile.intensity, 1)
        if intensity.max() > 0:
            intensity = (intensity / 65535).astype(np.float32)
        else:
            intensity = np.zeros((xyz.shape[0], 0), dtype=np.float32)

        try:
            rgb = np.concatenate(
                (np.expand_dims(lasFile.red, 1), np.expand_dims(lasFile.green, 1), np.expand_dims(lasFile.blue, 1)),
                1,
            )
            assert rgb.max() > 0

            rgb = rgb / 65535
        except Exception:
            rgb = np.zeros((xyz.shape[0], 0), dtype=np.float32)

        try:
            lbl = np.expand_dims(lasFile.classification, 1)
        except Exception:
            lbl = []
        
        data = np.concatenate([xyz, intensity, rgb, lbl], axis=-1)
    elif(file.endswith(".npy")):
        data = np.load(file, allow_pickle=False)
    elif(file.endswith(".npz")):
        data = np.load(path)
        ptsfts = data['arr_0']
        lbs = data['arr_1'].astype(np.float32)
        data = np.concatenate([ptsfts, lbs], axis=-1)
    else:
        print(f"{file} extension is not recognized")

    return data

def GetSampleCenters(pts, lbs, sampleCount, balanceClasses, blocksize, oneClass = None):
    randomCenters = []
    
    if not (oneClass is None):
        randomCenters = pts[np.random.choice(np.where(lbs == oneClass)[0], sampleCount)]
    elif balanceClasses:
        fileClasses = np.unique(lbs)
        classSampleCount = np.histogram(np.arange(sampleCount, dtype=int), bins=len(fileClasses))[0]
        randomCenters = np.concatenate([pts[np.random.choice(np.where(lbs == cls)[0], count)] for cls, count in zip(fileClasses,classSampleCount)])
    else:
        # pick a random points
        randomCenters = pts[np.random.randint(0, pts.shape[0]-1, sampleCount)]
        
    # ofcenter wanted classes to test if predictions are accurate with only part of the object
    jitter = np.random.uniform(-blocksize/2, blocksize/2, randomCenters.shape)
    randomCenters += jitter
        
    return randomCenters

def PickPointsInBlock(center, blocksize, pts, fts = None, lbs = None, npoints = None, selectRandomPoints = True):        
    # create the mask
    mask_x = np.logical_and(pts[:,0]<center[0]+blocksize/2, pts[:,0]>center[0]-blocksize/2)
    mask_y = np.logical_and(pts[:,1]<center[1]+blocksize/2, pts[:,1]>center[1]-blocksize/2)
    indices = np.where(np.logical_and(mask_x, mask_y))[0]
    
    if(len(indices) == 0):
        indices = np.arange(0, len(pts), 1)
        
    # DataTool().VisualizePointCloudAsync([pts[indices]])
    
    # random selection
    if not(npoints is None):        
        choice = RandomChoice(len(indices), npoints, selectRandomPoints)
        indices = indices[choice]
    
    pts = pts[indices]
    
    # DataTool().VisualizePointCloudAsync([pts])
    
    if not lbs is None:
        lbs = lbs[indices]
        
    if not fts is None:
        fts = fts[indices]
        
    return pts, fts, lbs

def GetMultipleTiles(pts, fts, lbs, center, input_tile_count, blocksize, npoints, selectRandomPoints):
    tile_count_widht = sqrt(input_tile_count)
    assert(tile_count_widht.is_integer())
    
    length = blocksize*(tile_count_widht - 1)
    tile_centers = np.array([[x, y, 0] for x in range(int(tile_count_widht)) for y in range(int(tile_count_widht))]) * blocksize - length/2
    tile_centers += center
    
    bigTilepts, bigTilefts, bigTilelbs = PickPointsInBlock(center, blocksize * tile_count_widht, pts, fts, lbs)                               
    data = [PickPointsInBlock(ctr, blocksize, bigTilepts, bigTilefts, bigTilelbs, npoints, selectRandomPoints) for ctr in tile_centers]
    
    temppts = np.stack([points for points, features, labels in data])
    tempfts = np.stack([features for points, features, labels in data])
    templbs = np.stack([labels for points, features, labels in data])
    
    visualize = False
    if visualize:
        boxes = [BoundingBoxFromVoxel(pt, blocksize) for pt in tile_centers + [0,0,blocksize]]
        # DataTool().VisualizePointCloudAsync([bigTilepts])
        DataTool().VisualizePointCloudAsync(list(temppts), bBoxes=boxes)
        DataTool().VisualizePointCloudAsync([pts], bBoxes=boxes)
    
    return temppts, tempfts, templbs

def ProcessSample(pts, fts, lbs, center, classCount, blocksize, npoints, selectRandomPoints, input_tile_count = 1):
    temppts, tempfts, templbs = CutPoints(pts, fts, lbs, center, blocksize, npoints, selectRandomPoints, input_tile_count)    
    templbs = HotEncodeLabels(templbs, classCount)
    return temppts, tempfts, templbs

def CutPoints(pts, fts, lbs, center, blocksize, npoints, selectRandomPoints, input_tile_count = 1):
    if input_tile_count == 1:
        temppts, tempfts, templbs = PickPointsInBlock(center, blocksize, pts, fts, lbs, npoints, selectRandomPoints)
    elif input_tile_count > 1:
        temppts, tempfts, templbs = GetMultipleTiles(pts, fts, lbs, center, input_tile_count, blocksize, npoints, selectRandomPoints)
    
    return temppts, tempfts, templbs

def HotEncodeLabels(templbs, classCount):
    # one hot encode labels
    if(len(templbs.shape) > 1):
        templbs = templbs.flatten()
    assert templbs.max() < classCount, f"templbs.max() < classCount is not True. templbs.max() = {templbs.max()}"
    templbs = np.eye(classCount, dtype=np.uint8)[templbs]
    
    return templbs

def AugmentSample(pts, fts, Mirror, Rotate, Scale, Jitter, FtrAugment):
    # data augmentation
    if(Mirror):
        pts = MirrorPoints(pts)
    if(Rotate):
        pts = RotatePointCloud(pts)
    if(Scale):
        pts = ScalePoints(pts, sigma = 0.02)
    if(Jitter):
        pts = JitterPoints(pts, sigma = 0.01)

    if(not (fts is None) and FtrAugment):
        if(fts.shape[-1] == 3):
            fts = JitterRGB(fts)
        elif(fts.shape[-1] == 1):
            fts = JitterReflectance(fts)
            
    return pts, fts

def ProcessFeatures(cts, data):
    # get the features
    fts = None
    lbs = None
    if(cts.featureComponents == 1):
        if not cts.noFeature:
            fts = np.expand_dims(data[:,3], 1).astype(np.float32)
        lbs = data[:,4].astype(int)
    else:
        if not cts.noFeature:
            fts = data[:,3:6].astype(np.float32)
        if data.shape[1] == 4:
            lbs = data[:,3].astype(int)
        else:
            lbs = data[:,-1].astype(int)
            
    pts, fts, lbs = cts.NormalizeData(data[:, :3], fts, lbs)
    
    assert lbs.min() == 0
    
    return pts, fts, lbs

def ReadLabels(file, verbose = False, readFormat=None):
    if(verbose):
        print("Reading labels of '{}'".format(file))

    t=time()

    lbl = None

    if(file.endswith(".hdf5") or readFormat == ".hdf5"):
        h5File = h5py.File(file, 'r')
        lbl = np.array(h5File["labels"])
        h5File.close()
    elif(file.endswith(".las") or file.endswith(".laz") or readFormat == ".las"):
        with laspy.open(file) as lasFile:
            lasFile = lasFile.read()
        lbl = lasFile.classification
        if(np.all(lbl == 0)):
            lbl = None
        
    elif(file.endswith(".labels") or file.endswith(".txt") or readFormat == ".txt" or readFormat == ".labels"):
        lbl = np.array(pd.read_csv(file, dtype=np.int8, header=None))
    elif(file.endswith(".ply") or readFormat == ".ply"):
        plydata = PlyData.read(file)
        lbl = plydata["vertex"].data["class"].astype(np.float32)
    elif(file.endswith(".npy") or readFormat == ".npy"):
        pc = np.load(file)        
        if(len(pc.shape) == 1):
            lbl = pc
        if(pc.shape[1] == 1):
            return pc
        elif(pc.shape[1] == 7):
            lbl = pc[:, 6]
        elif(pc.shape[1] == 5):
            lbl = pc[:, 4]
        elif(pc.shape[1] == 4):
            lbl = pc[:, 3]
        lbl = np.expand_dims(lbl, 1)
    elif(extension == ".ply"):
        plydata = PlyData.read(file)
        if("class" in plydata["vertex"].data.dtype.names):
            lbl = plydata["vertex"].data["class"].astype(np.float32)
            lbl = np.expand_dims(lbl, 1)

    if(lbl is None):
        return None

    if(len(lbl.shape) == 1):
        lbl = np.expand_dims(lbl, 1)

    if(verbose):
        print("Finished reading labels in {:.2f} min. Shape = {}".format((time() - t)/60, lbl.shape))

    return lbl

def ReadXYZL(file, lblFile = None, verbose = False):
    if(verbose):
        printline("Reading: '{}'".format(os.path.basename(file)))

    t=time()

    xyz = None
    lbl = None

    if(file.endswith(".hdf5")):
        h5File = h5py.File(file, 'r')
        pc = h5File["pointcloud"]
        xyz = pc[:, :3]
        if(lblFile is None):
            lbl = h5File["labels"]
        h5File.close()
    elif(file.endswith(".las") or file.endswith(".laz")):
        lasFile = laspy.file.File(file, mode = "r")
        xyz = np.concatenate((np.expand_dims(lasFile.x,1), np.expand_dims(lasFile.y,1), np.expand_dims(lasFile.z,1)), 1)
        xyz -= xyz.min(axis=0)
        lbl = lasFile.classification
        
    elif(file.endswith(".ply")):
        plydata = PlyData.read(file)
        x = plydata["vertex"].data["x"].astype(np.float32)
        y = plydata["vertex"].data["y"].astype(np.float32)
        z = plydata["vertex"].data["z"].astype(np.float32)
        xyz = np.concatenate((np.expand_dims(x,1), np.expand_dims(y,1), np.expand_dims(z,1)), axis=1)
        lbl = plydata["vertex"].data["class"].astype(np.float32)
    
    if(not (lblFile is None) and lblFile.endswith(".labels")):
        lbl = ReadLabels(lblFile)

    if(len(lbl.shape) == 1):
        lbl = np.expand_dims(lbl, 1)
    
    xyzl = np.concatenate((xyz, lbl), 1)
    
    printline("Finished in {:.2f} min. Shape = {}".format((time() - t)/60, xyzl.shape))
    return xyzl
    
def ReadHDF5Boxes(path):        
    h5File = h5py.File(path, 'r')
    boxesPos = np.array(h5File["boxes"])
    boundindBoxes = []

    for vox in boxesPos:
        boundindBoxes.append(BoundingBoxFromVoxel(Point(vox[0], vox[1], vox[2]), Const.voxelSize))

    return boundindBoxes

def ConvertToLasFolder(folder, outputFolder):
    pcFiles = [f for f in listdir(folder) if isfile(join(folder, f))]
    
    os.makedirs(outputFolder, exist_ok=True)
    for file in pcFiles:
        inputFile = join(folder, file)
        outputFile = join(outputFolder, os.path.splitext(file)[0]+".las")

        ConvertToLas(inputFile, outputFile)

def ConvertToLas(inputFile, outputFile):
    xyz = ReadXYZ(inputFile)
    rgb = ReadRGB(inputFile)
    lbl = ReadLabels(inputFile)

    SaveToLas(outputFile, xyz, rgb = rgb, labels = lbl)    

def SavePointCloud(path, xyz, intensity = None, rgb = None, labels = None, scale_rgb_intensity = 65535):
    name, extension = os.path.splitext(path)    
    if(extension == ".npy"):
        pc = xyz
        if not (intensity is None):
            if(len(intensity.shape) == 1):
                intensity = np.expand_dims(intensity * scale_rgb_intensity, axis=1)
            pc = np.concatenate([pc, intensity], axis=1)
        if not (rgb is None):
            pc = np.concatenate([pc, rgb * scale_rgb_intensity], axis=1)
        if not (labels is None):
            if(len(labels.shape) == 1):
                labels = np.expand_dims(labels, axis=1)
            pc = np.concatenate([pc, labels], axis=1)
        
        pc = pc.astype(np.float32)
        
        if(extension == ".npy"):
            np.save(path, pc, allow_pickle=False)
    elif(extension == ".npz"):
        np.savez_compressed(path, pts = xyz.astype(np.float32), intensity = (intensity*scale_rgb_intensity).astype(np.uint16), labels = labels.flatten().astype(np.uint8), allow_pickle=False)            
    elif(extension == ".las" or extension == ".laz"):
        SaveToLas(path, xyz, intensity, rgb, labels, scale_rgb_intensity, compress= (extension == ".laz"))

def SaveToLaz(file_path, xyz, intensity = None, rgb = None, labels = None, scale_rgb_intensity = 65535):
    SaveToLas(file_path, xyz, intensity, rgb, labels, scale_rgb_intensity, compress = True)

def SaveToLas(file_path, xyz, intensity = None, rgb = None, labels = None, scale_rgb_intensity = 65535, compress = False):
    """
    Expected pointcloud
    """
    # point_format=2
    # X	    X (x for scaled)	long[1] (4)
    # Y	    Y (y for scaled)	long[1] (4)
    # Z	    Z (z for scaled)	long[1] (4)
    # Intensity	intensity	unsigned short[1] (2)
    # (Flag Byte)	flag_byte	unsigned byte[1] (1)
    # (Classification Byte)	raw_classification	unsigned byte[1] (1)
    # User Data	user_data	unsigned char[1] (1)
    # Point Source Id	pt_src_id	unsigned short[1] (2)
    # Red	red	unsigned short[1] (2)
    # Green	green	unsigned short[1] (2)
    # Blue	blue	unsigned short[1] (2)
    new_las = laspy.create(file_version="1.4", point_format=8)
    new_las.header.are_points_compressed = compress
    new_las.header.offsets = list(np.floor(xyz.min(0)[:3]))
    new_las.header.scales = [
        0.001,
        0.001,
        0.001,
    ]  # scale lets you save decimal places. in this case 3 values after decimal point

    xyz = (xyz[:, :3] - np.array(new_las.header.offsets)) / np.array(new_las.header.scales)
    xyz = xyz.astype(np.float64)

    new_las.X = xyz[:, 0]
    new_las.Y = xyz[:, 1]
    new_las.Z = xyz[:, 2]

    if intensity is not None:
        new_las.intensity = (intensity * scale_rgb_intensity).flatten().astype(np.uint16)

    if rgb is not None:
        rgb = rgb * scale_rgb_intensity
        new_las.red = rgb[:, 0].astype(np.uint16)
        new_las.green = rgb[:, 1].astype(np.uint16)
        new_las.blue = rgb[:, 2].astype(np.uint16)

    if labels is not None:
        if labels.max() > 31:
            assert f"labels.max() == {labels.max()}. Las 1.4 supports up to 31 class number"

        new_las.classification = labels.flatten().astype(np.uint8)  # ubyte == uint8

    new_las.write(file_path)

class DataReader:
    threads = []
    dataset = []

    def ReadFiles(self, files, pointsDataSet = "points", silent=True, positionData = False):        
        if(type(files) is not list):
            files = [files]

        points = []
        labels = []
        position = []

        t=time()

        count = 0
        for f in files:            
            count+=1

            h5File = h5py.File(f, 'r')

            tempLabels = np.asarray(h5File["labels"], dtype="int8")
            if(tempLabels.shape[1] == 1):
                tempLabels = np.eye(Const.numOfCategories, dtype="int8")[tempLabels]
                tempLabels = np.squeeze(tempLabels, axis=2)

            if(len(points) == 0):
                points = np.asarray(h5File[pointsDataSet], dtype="float32")
                labels = tempLabels
                if(positionData):
                    position = np.asarray(h5File["position"], dtype="float32")
            else:
                points = np.concatenate((points, np.asarray(h5File[pointsDataSet], dtype="float32")))
                labels = np.concatenate((labels, tempLabels))
                if(positionData):
                    position = np.concatenate((position, np.asarray(h5File["position"], dtype="float32")))
            
            if(not silent):    
                print("Read file {}/{}. Voxels got: {}.".format(count, len(files), len(points)))
        
        if(not silent):
            elapsed = round(time() - t)
            print("{} dataset read in {:.0f} min {:.0f} sec".format(len(files), (elapsed - (elapsed % 60))/60, elapsed % 60))

        if(positionData):
            return points, position, labels
        else:
            return points, labels

class Point:
    def __init__(self, x, y, z, label = -1):
        self.x = x
        self.y = y
        self.z = z
        self.label = label
    
    @staticmethod
    def from_XYZL(XYZL):
        return Point(XYZL[0], XYZL[1], XYZL[2], XYZL[3])
    
    @staticmethod
    def from_XYZ(XYZ):
        return Point(XYZ[0], XYZ[1], XYZ[2])

def GetPointsInBoundingBox(points, boundingBox):
    if(len(boundingBox) != 6):
        return None

    rows = GetPointsIndexInBoundingBox(points, boundingBox)

    return points[rows]

def RotatePointCloud(batch_data):
    assert(batch_data.shape[-1]==3)
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
    assert(points.shape[-1]==3)

    scale = np.random.uniform(1-sigma, 1+sigma)
    scale_matrix = np.array([[scale, 0, 0],
                             [0, scale, 0],
                             [0, 0, scale]])
    scaled = np.dot(points, scale_matrix)

    return scaled

def RandomChoice(length : int, n : int, replace = False):
    if replace:
        return np.random.choice(length, n, replace=True) 
    else:
        if(length < n):
            indices = np.concatenate([np.arange(length), np.random.choice(length, n-length, replace=True)], axis=0)
            return indices[np.random.choice(n, n, replace=False)]
        else:
            return np.random.choice(length, n, replace=False)

def CountPointsInBox(points, boundingBox):
    if(len(boundingBox) != 6):
        return None
    
    indices = GetPointsIndexInBoundingBox(points, boundingBox)
    return len(indices[0])

def GetPointsIndexInBoundingBox(points, boundingBox):
    if(len(boundingBox) != 6):
        return None

    return np.where((points[:,0] >= boundingBox[0]) & (points[:,0] <= boundingBox[1]) &
                    (points[:,1] >= boundingBox[2]) & (points[:,1] <= boundingBox[3]) &
                    (points[:,2] >= boundingBox[4]) & (points[:,2] <= boundingBox[5]))

def BoundingBoxFromVoxel(vxlCntr, vxlEdge):
    if type(vxlEdge) is int or type(vxlEdge) is float or type(vxlEdge) is np.float64:
        subEdgeX = vxlEdge/2
        subEdgeY = vxlEdge/2
        subEdgeZ = vxlEdge/2
    elif(len(vxlEdge) == 3):
        subEdgeX = vxlEdge[0]/2
        subEdgeY = vxlEdge[1]/2
        subEdgeZ = vxlEdge[2]/2
    
    if isinstance(vxlCntr, Point):
        x = vxlCntr.x 
        y = vxlCntr.y
        z = vxlCntr.z
    else:
        x = vxlCntr[0]
        y = vxlCntr[1]
        z = vxlCntr[2]

    minX = x - subEdgeX
    maxX = x + subEdgeX
    
    minY = y - subEdgeY
    maxY = y + subEdgeY
        
    minZ = z - subEdgeZ
    maxZ = z + subEdgeZ

    return [minX, maxX, minY, maxY, minZ, maxZ]

def GetGlobalBoundingBox(points, discardZeros = False):
    
    if(discardZeros):
        points = np.array(points)
        indexes = np.where(points[:] == [0, 0, 0])
        points = np.delete(points, indexes, axis=0)

    mins = np.amin(points, axis = 0)
    maxs = np.amax(points, axis = 0)    

    return [mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]]

def hex_to_RGB(hex):
  ''' "#FFFFFF" -> [255,255,255] '''
  # Pass 16 to the integer function for change of base
  return [int(hex[i:i+2], 16) for i in range(1,6,2)]

def RGB_to_hex(RGB):
  ''' [255,255,255] -> "#FFFFFF" '''
  # Components need to be integers for hex to make sense
  RGB = [int(x) for x in RGB]
  return "#"+"".join(["0{0:x}".format(v) if v < 16 else
            "{0:x}".format(v) for v in RGB])

def LinearGradient(start_hex, finish_hex="#FFFFFF", n=10):
    ''' returns a gradient list of (n) colors between
        two hex colors. start_hex and finish_hex
        should be the full six-digit color string,
        inlcuding the number sign ("#FFFFFF") '''
    # Starting and ending colors in RGB form
    s = hex_to_RGB(start_hex)
    f = hex_to_RGB(finish_hex)
    # Initilize a list of the output colors with the starting color
    RGB_list = [s]
    # Calcuate a color at each evenly spaced value of t from 1 to n
    for t in range(1, n):
        # Interpolate RGB vector for color at the current value of t
        curr_vector = [
        int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
        for j in range(3)
        ]
        # Add it to our list of output colors
        RGB_list.append(curr_vector)

    return RGB_list

def SaveBoundingBoxes(file_path, bBoxes):
    file = open(file_path,"w")

    for box in bBoxes:
        file.write(str(box[0])+" "+str(box[1])+" "+str(box[2])+" "+str(box[3])+" "+str(box[4])+" "+str(box[5])+"\n")
    
    file.close()

def SaveVoxels(file_path, voxels):
    file = open(file_path,"w")

    for vox in voxels:
        file.write(str(vox[0])+" "+str(vox[1])+" "+str(vox[2])+" "+str(vox[3])+" "+str(vox[4])+" "+str(vox[5])+" "+str(vox[6])+"\n")
    
    file.close()

def ReadBoundingBoxes(file_path):
    file = open(file_path,"r")
    boundingBoxes = []

    for line in file:
        fl = line.split()

        floats = []
        for l in fl:
            floats.append(float(l))

        boundingBoxes.append(floats)
    
    file.close()
    return boundingBoxes

def DownsampleAndAddclass(points, classNum, voxelSize = -1):
    if(voxelSize != -1):
        pointCloud = o3d.geometry.PointCloud()
        pointCloud.points = o3d.utility.Vector3dVector(points)    
        pointCloud = o3d.geometry.voxel_down_sample(pointCloud, voxel_size=voxelSize)
        points = np.asarray(pointCloud.points)

    labels = np.full((len(points), 1), classNum)
    points = np.append(points, labels, axis = 1)

    return points

def PrepPointCloud(dataIN, objectLabel, noObjectLabel, downSampleVoxel = -1, verbose = False):
    dataTool = DataTool()
    print("Reading: {}".format(dataIN))
    worldPoints = ReadXYZ(dataIN)
    pointLabels = ReadLabels(dataIN)

    indexes = np.nonzero(pointLabels == Label.cars)
    carPoints = worldPoints[indexes]
    worldPoints = np.delete(worldPoints, indexes, axis=0)

    carPoints = DownsampleAndAddclass(carPoints, objectLabel, downSampleVoxel)
    worldPoints = DownsampleAndAddclass(worldPoints, noObjectLabel, downSampleVoxel)

    pointCloud = np.concatenate((carPoints, worldPoints))
    #pointCloud = carPoints
    if(verbose):
        print("Points left: {}".format(len(pointCloud)))

    return pointCloud

def FilterSpecClassVoxels(voxels, classLabel, noClassLabel, minPointCount = 0):
    accepted = []
    rejected = []

    avgPointsCountInVoxel = 0
    avgClassPointCount = 0

    for vox in voxels:
        points = np.array(vox)
        indexes = np.where(points[:,3] == classLabel)[0]

        avgPointsCountInVoxel += len(vox)
        avgClassPointCount += len(indexes)

        if(len(indexes) >= minPointCount):
            accepted.append(vox)
        else:
            rejected.append(vox)
    
    return np.array(accepted), np.array(rejected)

def GetBoundingBoxesOfPoint(voxels, verbose = False, discardZeros = True):
    boxes = []

    maxx = 0
    maxy = 0
    maxz = 0    

    for vox in voxels:
        box = GetGlobalBoundingBox(vox, discardZeros)
        boxes.append(box)

        if(verbose):
            maxx = max(maxx, box[1]-box[0])
            maxy = max(maxy, box[3]-box[2])
            maxz = max(maxz, box[5]-box[4])

    if(verbose):
        print("max x len: {}, max y len: {}, max z len: {}".format(maxx, maxy, maxz))

    return boxes

def AddNoise(batch_data):    

    return batch_data.shape

def GetPointsAndLabels(voxels, numOfPointInSample):
    points = []
    labels = []

    for vox in voxels:                    
        if(len(vox) < numOfPointInSample):
            zeros = np.zeros((numOfPointInSample-len(vox), 4))
            vox = np.concatenate((vox, zeros))
        
        #shuffle
        indexes = np.random.choice(len(vox), numOfPointInSample, replace = False)
        vox = vox[indexes]

        points.append(vox[..., 0:3])
        labels.append(vox[..., 3])    

    return np.array(points), np.array(labels)

def GetPointsAndAns(voxels, numOfPointInSample, classLabel, minCountOfTrue = 100):
    points = []
    labels = np.zeros(shape=(len(voxels), 2), dtype = np.float32)

    avgPointsInVoxel = 0
    acceptedCarVoxels = 0
    avgCarPointsInVoxels = 0
    i = 0
    for vox in voxels:                   
        avgPointsInVoxel += len(vox)

        res = np.where(vox[..., 3] == classLabel)
        if(len(res[0]) >= minCountOfTrue):
            labels[i] = [1.0, 0.0]
            avgCarPointsInVoxels += len(res[0])
            acceptedCarVoxels += 1
        else:
            labels[i] = [0.0, 1.0]

        if(len(vox) < numOfPointInSample):
            zeros = np.zeros((numOfPointInSample-len(vox), 4))
            vox = np.concatenate((vox, zeros))
        
        #shuffle
        indexes = np.random.choice(len(vox), numOfPointInSample, replace = False)
        points.append(vox[indexes, 0:3])
        
        i+=1

    avgPointsInVoxel /= len(voxels)
    avgCarPointsInVoxels /= acceptedCarVoxels
    acceptedCarVoxels /= len(voxels)

    return np.array(points), labels, avgPointsInVoxel, acceptedCarVoxels, avgCarPointsInVoxels

def CreateCheckPointFile():
    open(Paths.checkPointFilePath, "w")

def CreatePausePointFile():
    open(Paths.pausePointFilePath, "w")

def DeleteCheckPointFile():
    if(IsCheckPointFileExists()):
        remove(Paths.checkPointFilePath)

def DeletePausePointFile():
    remove(Paths.pausePointFilePath)

def IsCheckPointFileExists():
    return isfile(Paths.checkPointFilePath)

def IsPausePointFileExists():
    if(not isfile(Paths.pausePointFilePath)):
        print("Press enter to continue")
        input()
        CreatePausePointFile()

def modelPath():
    if(not exists(Paths.dataPath)):
        return None

    pcFiles = [Paths.dataPath+"/"+f for f in listdir(Paths.dataPath) if isfile(join(Paths.dataPath, f)) and f.startswith('model')]
    
    if(len(pcFiles) == 0):
        return None

    assert len(pcFiles) == 1, "More than one model in data folder"   
    return pcFiles[0]

def FireBaseStuff():
    # def writeMessageToFirebase():
    #     from firebase import Firebase
    #     config = {
    #         "apiKey": "AIzaSyDytF5CdxpfgBr3YlZ86CcSVKt_z3mRJbU",
    #         "authDomain": "online-app-600.firebaseapp.com",
    #         "databaseURL": "https://online-app-600.firebaseio.com",
    #         "storageBucket": "online-app-600.appspot.com"
    #     }
    #     firebase = Firebase(config)

    #     auth = firebase.auth()
    #     #auth.create_user_with_email_and_password("jonukas555@gmail.com", "rootmeROOTME")
    #     #user = auth.sign_in_with_email_and_password("jonukas555@gmail.com", "rootmeROOTME")
    #     user = auth.sign_in_with_email_and_password("jonasstankevicius.js@gmail.com", "testTEST")
    #     print(auth.get_account_info(user['idToken']))
    #     #auth.send_email_verification(user['idToken'])

    #     db = firebase.database()
    #     data = db.child("duomenys").get()

    #     #storage = firebase.storage()
    #     #data = storage.child("images/google-services.json").get_url(None)

    #     print(data.val())

    # def readMessagesFromFirestore():
    #     users_ref = firestoreDB.collection('duomenys')
    #     docs = users_ref.stream()

    #     for doc in docs:
    #         data = doc.to_dict()
    #         print('{} => {}'.format(data["name"], data["text"]))

    # def writeMessageToFirestore():
    #     newDocKey = firestoreDB.collection('duomenys').document();

    #     doc_ref = firestoreDB.collection('duomenys').document(newDocKey.id)

    #     doc_ref.set({
    #         'name': 'Jonas',
    #         'test': 'works from python',
    #     })
    return 0

def PostMessage(dictData, training):    
    message = ""

    for key, value in dictData.items():
        if(type(value) is float):
            message += "{}:{:.3f}. ".format(str(key), value)
        else:
            message += "{}:{}. ".format(str(key), value)

    print(message)

    if(training):
        file = open(Paths.trainLogPath, "a")
        #newDocRef = firestoreDB.collection(FireStroreCollection.train).document()
        #col_ref = firestoreDB.collection(FireStroreCollection.train).document(newDocRef.id)
    else:
        file = open(Paths.dataProcPath, "a")
        #newDocRef = firestoreDB.collection(FireStroreCollection.dataProc).document()
        #col_ref = firestoreDB.collection(FireStroreCollection.dataProc).document(newDocRef.id)
    
    file.write(message+"\n")

    try:
        notifyDevice.send(message)
        #col_ref.set(dictData)
    except:
        print("Online message error")

def UpSampleBatchSize(points, labels, numOfPoints):
    points = np.array(points)
    labels = np.array(labels)

    newPoint = []
    newLabels = []
    
    for batch, labels in zip(points, labels):
        zeros = np.zeros((numOfPoints-len(batch), len(batch[0])))
        batch = np.concatenate((batch, zeros))

        zeros = np.zeros((numOfPoints-len(batch), len(labels[0])))
        labels = np.concatenate((labels, zeros))

        indexes = np.random.choice(len(points[0]), numOfPoints, replace = False)

        newPoint.append(batch[indexes])
        newLabels.append(labels[indexes])

    return np.array(newPoint), np.array(newLabels)

class memoryCheck():
    """Checks memory of a given system""" 
    def __init__(self): 
        if os.name == "posix":
            self.value = self.linuxRam()
        elif os.name == "nt":
            self.value = self.windowsRam()
 
    def windowsRam(self):
        """Uses Windows API to check RAM in this OS"""
        kernel32 = ctypes.windll.kernel32
        c_ulong = ctypes.c_ulong
        class MEMORYSTATUS(ctypes.Structure):
            _fields_ = [
                ("dwLength", c_ulong),
                ("dwMemoryLoad", c_ulong),
                ("dwTotalPhys", c_ulong),
                ("dwAvailPhys", c_ulong),
                ("dwTotalPageFile", c_ulong),
                ("dwAvailPageFile", c_ulong),
                ("dwTotalVirtual", c_ulong),
                ("dwAvailVirtual", c_ulong)
            ]
        memoryStatus = MEMORYSTATUS()
        memoryStatus.dwLength = ctypes.sizeof(MEMORYSTATUS)
        kernel32.GlobalMemoryStatus(ctypes.byref(memoryStatus))
 
        return int(memoryStatus.dwTotalPhys/1024**2)
 
    def linuxRam(self):
        """Returns the RAM of a linux system"""
        totalMemory = os.popen("free -m").readlines()[1].split()[1]
        return int(totalMemory)

def GetLabelPoint(points, label):
    indexes = np.where(points[:,3] == label)
    return points[indexes], indexes

def VisualizeDataset(dataPath, markEachClass = False, pointsDataSet = "points", hasBoxesData = False):    
    print("Visualize Dataset")
    points = []
    downSample = False
    dataTool = DataTool()

    if(pointsDataSet == "points"):
        h5File = h5py.File(dataPath, 'r')
        pointcloud = np.array(np.asarray(h5File[pointsDataSet]))
        pointsCount = len(pointcloud)*len(pointcloud[0])
        pointcloud = pointcloud.reshape(pointsCount, pointcloud.shape[2])

        if(markEachClass):
            labels = np.array(np.asarray(h5File["labels"], dtype="int8"))
            if(len(labels[0,0]) == 9):
                labels = labels.reshape(pointsCount, 9)
                labels = np.argmax(labels, axis=1)
            else:
                labels = labels.reshape(pointsCount, 1)
            labels = np.expand_dims(labels, 1)
            pointcloud = np.append(pointcloud, labels, axis=1)

        indexes = np.where(pointcloud[:, :3] == [0., 0., 0.])
        points = np.delete(pointcloud, indexes, axis=0)

        print("Nonzero values {:.3f} => {}".format(len(points)/pointsCount, len(points)))
        print(GetGlobalBoundingBox(points))
    else:
        points = ReadXYZL(dataPath)
        downSample = True        

    boundingBoxes = []
    if(hasBoxesData):
        if(pointsDataSet == "points"):
            boundingBoxes = ReadHDF5Boxes(dataPath)
        else:
            boundingBoxes = ReadHDF5Boxes("G:/PointCloud DataSets/semenatic3d/processedTest/test_bildstein_station3_xyz_intensity_rgb.hdf5")

    if(markEachClass):
        print("Set colors for each label")

        # manMadeTerrain, naturalTerrain, highVegetation, lowVegetation, buildings, hardScape, cars, unlabeled = SeparateEachClass(points)
        manMadeTerrain, naturalTerrain, highVegetation, lowVegetation, buildings, hardScape, cars = SeparateEachClass(points)

        dataTool.VisualizePointCloud(   [manMadeTerrain, naturalTerrain, highVegetation, lowVegetation, buildings, hardScape, cars], 
                                        [[1,0.5,0],     [0,1,0.5],          [0,1,0],      [0.2, 1, 0.3],  [0,0,1],  [0.5,0,1],  [1,0,0]], downSample, boundingBoxes)
    else:
        dataTool.VisualizePointCloud([points], [None], downSample, boundingBoxes)

def SeparateEachClass(points):    
    count = np.max(points[:,3])

    classesPoints = []

    for i in range(1, int(count)+1):
        pts, _ = GetLabelPoint(points, i)
        classesPoints.append(pts)  

    return classesPoints

def GetOffsetArray(pointCloud):    
    mins = [0,0,0]
    if(len(pointCloud.shape) == 2):
        mins = np.amin(pointCloud, axis=0)
    elif(len(pointCloud.shape) == 3):
        mins = np.amin(pointCloud, axis=1)
        assert False
    else:
        return 0
    
    minX = mins[0]
    minY = mins[1]
    minZ = mins[2]

    offset = np.array([minX, minY, minZ])
    for i in range(len(offset)):
        if(offset[i] > 0):
            offset[i] *= -1
        else:
            offset[i] = abs(offset[i])
    
    return offset

def printline(message):
    addSpaces = max(60 - len(message), 0)
    print(message, end = " " * addSpaces + "\r")

def VoxelizeDataset(pointCloud, WindowXYZ, MinPointCountInWindow, PointCountInWindow, PointComponents, labelCount = Label.Semantic3D.Count, discardPoints = True, returnVoxelIndexes = False):
    if(os.path.exists(pointCloud)):
        pointCloud = ReadXYZL(path)
    pointCloud = pointCloud.astype(np.float32)
    
    printline("Voxelizing {} points".format(len(pointCloud)))
    t=time()

    pointLabels = pointCloud[:, 3]
    pointCloud = np.delete(pointCloud, 3, 1)
    pointCloud = pointCloud + GetOffsetArray(pointCloud)
    voxel_index = np.floor(pointCloud[:] / WindowXYZ).astype(np.int32)

    indices = np.expand_dims(np.arange(len(voxel_index)), axis=1)
    voxel_index = np.concatenate((voxel_index, indices), axis=1)
    
    printline("Sorting points")
    voxel_index = voxel_index[np.lexsort((voxel_index[:,2], voxel_index[:,1], voxel_index[:,0]))]
    pointCloud = np.concatenate((pointCloud[voxel_index[:,3]], np.expand_dims(pointLabels[voxel_index[:,3]], axis=1)), axis=1)
    voxel_index = np.delete(voxel_index, 3, 1)

    printline("Dividing points into voxels") 
    uniqueVoxels, indexes, counts = np.unique(voxel_index, axis=0, return_counts= True, return_index=True)

    indexes = indexes[1:]
    pointCloud = np.split(pointCloud, indexes, axis=0)
    # pointLabels = np.split(pointLabels, indexes, axis=0)

    if(discardPoints):
        delIndexes = np.where(counts[:] < MinPointCountInWindow)
        pointCloud = np.delete(pointCloud, delIndexes, axis=0)
        # pointLabels = np.delete(pointLabels, delIndexes, axis=0)
        counts = np.delete(counts, delIndexes, axis=0)

    goodVoxels = np.where(counts[:] >= PointCountInWindow)
    choices = [np.random.choice(i, PointCountInWindow, replace = False) for i in counts[goodVoxels]]

    points = np.zeros((len(pointCloud), PointCountInWindow, PointComponents))
    labels = np.zeros((len(pointCloud), PointCountInWindow, labelCount))

    printline("Generating fullsize voxels")
    goodVoxels = goodVoxels[0]
    for i in range(len(goodVoxels)):
        data = np.array(pointCloud[goodVoxels[i]], dtype=np.float32)
        data = data[choices[i]]

        pts = data[:,:3]
        lbl = data[:,3]
        lbl = np.eye(labelCount)[lbl.astype(np.int32)] #one hot encode

        points[goodVoxels[i]] = np.expand_dims(pts, axis=0)
        labels[goodVoxels[i]] = np.expand_dims(lbl, axis=0)
    
    printline("Generating smaller voxels")
    badVoxels = np.delete(np.arange(len(pointCloud)), goodVoxels, axis=0)
    for i in range(len(badVoxels)):
        data = np.array(pointCloud[badVoxels[i]], dtype=np.float32)
    
        zeros = np.zeros((PointCountInWindow-len(data), PointComponents+1), dtype=np.float32)
        data = np.concatenate((data, zeros))    
        np.random.shuffle(data) # I think it is a good practice to randomize training data

        pts = data[:,:3]
        lbl = data[:,3]
        lbl = np.eye(labelCount)[lbl.astype(np.int32)] #one hot encode

        points[badVoxels[i]] = np.expand_dims(pts, axis=0)
        labels[badVoxels[i]] = np.expand_dims(lbl, axis=0)

    print("{} voxels generated in {:.3f}".format(len(points), (time() - t)/60))

    if(returnVoxelIndexes):
        return points, labels, uniqueVoxels
    else:
        return points, labels

def Recompress():
    pcFiles, testFile = Paths.GetRawFiles()
    pcFiles.append(testFile)

    for file in pcFiles:
        pointcloud = ReadXYZ(file)
        labels = ReadLabels(file)

        pointcloud = np.concatenate((pointcloud, np.expand_dims(labels,1)), axis=1)
        pointcloud = pointcloud[np.lexsort((pointcloud[:,2], pointcloud[:,1], pointcloud[:,0]))]
        
        labels = pointcloud[:, 3]
        pointcloud = np.delete(pointcloud, 3, 1)
        pointcloud = pointcloud + GetOffsetArray(pointcloud)

        output_path = join(Paths.trainDataSetsPath, os.path.basename(file))

        h5File = h5py.File(output_path, 'w')        
        h5File.create_dataset("pointcloud", data=pointcloud, dtype='float32', compression="lzf")
        h5File.create_dataset("labels", data=labels, dtype='uint8', compression="lzf")
        h5File.create_dataset("shape", data=np.array(pointcloud.shape), dtype='int64', compression="lzf")
        h5File.create_dataset("boundingbox", data=GetGlobalBoundingBox(pointcloud), dtype='int64', compression="lzf")
        h5File.close()

def ConvertTxtToH5(folder, outputFolder):
    pcFiles = [os.path.splitext(os.path.basename(f))[0] for f in listdir(folder)
                if isfile(join(folder, f))
                and f.endswith('.txt')]

    print("reading {} files".format(len(pcFiles)))
    for file in pcFiles:
        print("Reading: {}".format(file))
        t = time()

        pointcloud = np.loadtxt(os.path.join(folder, file+".txt")).astype(np.float16)

        # h5File = h5py.File(os.path.join(outputFolder, file+".hdf5"), 'w')
        # h5File.create_dataset("pointcloud", data=pointcloud[:,:3], dtype='float32', compression="lzf")
        # h5File.create_dataset("intensity_rgb", data=pointcloud[:,3:], dtype='int16', compression="lzf")        

        labelsFile = os.path.join(folder, file+".labels")
        if os.path.isfile(labelsFile):
            labels = np.loadtxt(labelsFile).astype(np.int8)
            pointcloud = np.concatenate((pointcloud, np.expand_dims(labels, 1)), axis=1)
            # h5File.create_dataset("labels", data=labels, dtype='int8', compression="lzf")
            del labels

        np.save(os.path.join(outputFolder, file), pointcloud, allow_pickle=False)
        print(pointcloud.shape)

        del pointcloud

        # h5File.close()
        print("done in {}:{} min.".format(int((time() - t)/60), int((time() - t)%60)))

def PreprocessDatasetToVoxels(folder, outputFolder, extension = ".hdf5", saveFormat = ".npy", override = False, voxelSize = 0.1, splitIntoChunks = False, chunkSize = 40, label_map = None):
    if(not os.path.isdir(outputFolder)):
        os.mkdir(outputFolder)

    pcFiles = [os.path.join(folder, f) for f in listdir(folder)
                if isfile(join(folder, f))
                and f.endswith(extension)]

    print("Found files: ",len(pcFiles))

    for file in pcFiles:
        t = time()
        targetFile = os.path.join(outputFolder, os.path.splitext(os.path.basename(file))[0]+saveFormat)
        if os.path.isfile(targetFile) and not override and not splitIntoChunks:
            print("Already exists: ",targetFile)
            continue
        else:
            print("Generating: ",targetFile)
            if(not splitIntoChunks):
                open(targetFile, "a").close()

        print("Reading: {}".format(file))
        if(file.endswith(".las") or file.endswith(".laz")):
            data = ReadXYZRGBLBL(file)
            if(data.shape[1] == 4):
                lbl = np.expand_dims(data[:,3], -1)
                pts = data[:,:3]
                fts = np.zeros((len(data), 1), dtype=np.float32)
            else:                
                if(data.shape[1] == 8):
                    splits = [3, 7]
                elif(data.shape[1] == 7):
                    splits = [3, 6]
                pts, fts, lbl = np.split(data, splits, axis=-1)
        else:
            pts = ReadXYZ(file)
            lbl = ReadLabels(file)
            fts = ReadRGB(file)

        shift = pts.min(axis=0)
        pts -= shift

        if(splitIntoChunks):
            discretized = ((pts[:,:2]).astype(float)/chunkSize).astype(int)
            chunkCenters = np.unique(discretized, axis=0)
            chunkCenters = chunkCenters.astype(np.float32) * chunkSize

            print(f"Split dataset into {len(chunkCenters)} chunks, each square size : {chunkSize}")

            for i, pt in tqdm(enumerate(chunkCenters)):
                mask_x = np.logical_and(pts[:,0]<pt[0]+chunkSize/2, pts[:,0]>pt[0]-chunkSize/2)
                mask_y = np.logical_and(pts[:,1]<pt[1]+chunkSize/2, pts[:,1]>pt[1]-chunkSize/2)
                mask = np.logical_and(mask_x, mask_y)
                chunkPts = pts[mask]
                chunkFts = fts[mask]
                chunkLbl = lbl[mask]

                if(len(chunkPts)):
                    targetFile = os.path.join(outputFolder, os.path.splitext(os.path.basename(file))[0]+f"_{i+1}.npy")
                    pc = Voxelize(chunkPts, chunkFts, chunkLbl, voxelSize = voxelSize, removeZeroLabels = False)
                    np.save(targetFile, pc, allow_pickle=False)
            
        elif(extension == ".ply"):
            squares = CutToSquares(pts, fts, lbl)

            for i, sq in enumerate(squares):
                newName = os.path.join(outputFolder, os.path.splitext(os.path.basename(file))[0]+f"_{i}.npy")
                if(os.path.isfile(newName) and not override):
                    continue

                np.save(newName, sq)
        else:
            print("Processing")
            
            if not(label_map is None):
                lbl = ChangeLabels(lbl, label_map)
                
            pc = Voxelize(pts, fts, lbl, voxelSize = voxelSize, removeZeroLabels = False)                    
            pc[:,:3] += shift
            
            if(pc.shape[1] == 4):
                SavePointCloud(targetFile, pc[:,:3], labels = pc[:,3])
            elif(pc.shape[1] == 5):                
                SavePointCloud(targetFile, pc[:,:3], intensity = pc[:,3], labels = pc[:,4])
            elif(pc.shape[1] == 7):                
                SavePointCloud(targetFile, pc[:,:3], rgb = pc[:,3:6], labels = pc[:,6])
            elif(pc.shape[1] == 8):                
                SavePointCloud(targetFile, pc[:,:3], rgb = pc[:,3:6], intensity = pc[:,6], labels = pc[:,7])
            else:
                assert "Unexpected output shape"
                
            print("done in {}:{} min.".format(int((time() - t)/60), int((time() - t)%60)))

def CutToSquares(pts, fts, lbl):
    if(lbl is None):
        pts = np.concatenate([pts, fts], axis = 1)
    else:
        pts = np.concatenate([pts, fts, lbl], axis = 1)

    pca = PCA(n_components=1)
    pca.fit(pts[::10,:2])
    pts_new = pca.transform(pts[:,:2])
    _, edges = np.histogram(pts_new, pts_new.shape[0]// 2500000)
    
    squares = []

    for i in range(1,edges.shape[0]):    
        mask = np.logical_and(pts_new<=edges[i], pts_new>edges[i-1])[:,0]
        batch = pts[mask]

        if(not (lbl is None)):
            index = np.where(batch[:,4] == 0)[0]
            batch = np.delete(batch, index, 0)

        squares.append(batch)

    _, edges = np.histogram(pts_new, pts_new.shape[0]// 2500000 -2, range=[(edges[1]+edges[0])//2,(edges[-1]+edges[-2])//2])

    for i in range(1,edges.shape[0]):
        mask = np.logical_and(pts_new<=edges[i], pts_new>edges[i-1])[:,0]
        batch = pts[mask]        
        
        if(not (lbl is None)):
            index = np.where(batch[:,4] == 0)[0]
            batch = np.delete(batch, index, 0)

        squares.append(batch)
    
    return squares

def Voxelize(pts, rgb, lbl = None, voxelSize = 0.1, removeZeroLabels = True):
    if(not (lbl is None) and removeZeroLabels):
        # print("Delete unlabeled points")
        delIndices = np.where(lbl[:] == 0)
        print("{} unlabeled points".format(len(delIndices[0])))
        pts = np.delete(pts, delIndices, axis=0)
        rgb = np.delete(rgb, delIndices, axis=0)
        lbl = np.delete(lbl, delIndices, axis=0)

    voxelSize = np.array([voxelSize, voxelSize, voxelSize])

    print("Preparing voxel indexes")
    voxel_index = ((np.floor(pts / voxelSize) * voxelSize) + voxelSize*0.5).astype(np.float32)

    # print("Adding distances to points")
    dist = pts - voxel_index
    # print("np.power")
    dist = np.power(dist, 2)
    # print("np.sum")
    dist = np.sum(dist, 1)
    # print("np.expand_dims")
    dist = np.expand_dims(dist, 1)

    # sort points
    # print("Adding indexes to voxel indexes")
    voxel_index = np.concatenate((voxel_index, np.expand_dims(np.arange(len(voxel_index)), axis=1)), axis=1)
    print("Sorting voxel indexes")
    voxel_index = voxel_index[np.lexsort((voxel_index[:,2], voxel_index[:,1], voxel_index[:,0]))]
    # print("Rearanging data")
    index = voxel_index[:, 3].astype(np.int32)
    pts = pts[index]
    rgb = rgb[index]
    if(not (lbl is None)):
        lbl = lbl[index]
    dist = dist[index]

    # print("Deleting from voxel indexes")
    voxel_index = voxel_index[:,:3]
    
    #split to voxels
    print("Doing GetSplitPoints")
    indexes = GetSplitPoints(voxel_index)
    # _, indexesTemp = np.unique(voxel_index, axis = 0, return_index= True) # to check all good

    # print("Doing np.split pts")
    pts = np.split(pts, indexes, axis=0)
    # print("Doing np.split rgb")
    rgb = np.split(rgb, indexes, axis=0)
    if(not (lbl is None)):
        # print("Doing np.split lbl")
        lbl = np.split(lbl, indexes, axis=0)
    # print("Doing np.split dist")
    dist = np.split(dist, indexes, axis=0)
    del indexes
    indexes = None

    print("Doing smart array loop")
    mins = [np.argmin(dist[i][:,0]) for i in range(len(pts))]    
    if(not (lbl is None)):
        data = np.array([np.concatenate((pts[i][mins[i]], rgb[i][mins[i]], lbl[i][mins[i]]), axis=0) for i in range(len(mins))])
    else:
        data = np.array([np.concatenate((pts[i][mins[i]], rgb[i][mins[i]]), axis=0) for i in range(len(mins))])

    data[:,:3] = np.floor(data[:,:3] / voxelSize) * voxelSize #floor xyz values
    return data

from numba import njit
@njit
def GetSplitPoints(voxel_index):
    indices = []

    for i in range(1, len(voxel_index)):
        if(not np.array_equal(voxel_index[i], voxel_index[i-1])):
            indices.append(i)
    
    return indices

def UpscaleFile(ptslbsFile, originalFile, outputFile):
    from dataTool import ReadLabels, ReadXYZ

    print("Upscaling: {}".format(ptslbsFile))
    scores = ReadLabels(ptslbsFile)
    scores = np.squeeze(scores, 1)
    pts_src = ReadXYZ(ptslbsFile)
    originalPoints = ReadXYZ(originalFile)

    UpscaleToOriginal(originalPoints, pts_src, scores, outputFile)

def UpscaleToOriginal(originalPoints, pts_src, lbl, outputFile = None):
    from tqdm import tqdm
    # create the scores for all points
    step = 10000000 #1000000

    if(len(lbl.shape) == 1):
        fullLbl = np.zeros((0,), np.int8)
    else:
        fullLbl = np.zeros((0,1), np.int8)
    
    print("Build KDTree")
    t = time()
    kdt = KDTree(pts_src, leaf_size=50)
    print("KDTree magic done in {}. Source pts: {}. Queary pts: {}".format(time() - t, len(pts_src), len(originalPoints)))
    for i in tqdm(range(0, ceil(len(originalPoints)/step))):
        a = i*step
        b = a + np.min([len(originalPoints)-a, step])

        _, indexes = kdt.query(originalPoints[a:b], k = 1)
        indexes = np.squeeze(indexes, 1)

        fullLbl = np.concatenate([fullLbl, lbl[indexes]], 0)

    if(not (outputFile is None)):
        if(outputFile.endswith(".las")):
            SaveToLas(outputFile, originalPoints, labels = fullLbl)
        else:
            SaveLabelsPnts(fullLbl, outputFile)
    else:
        return fullLbl
    
def nearest_correspondance(pts_src, pts_dest, K=1):
    # print("KDTree magic. Source pts: {}. Queary pts: {}".format(len(pts_src), len(pts_dest)))
    # t = time()
    kdt = KDTree(pts_src, leaf_size=20)
    _, indexes = kdt.query(pts_dest, k = K)
    # print("Done in {}:{} min.".format(int((time() - t)/60), int((time() - t)%60)))    
    return np.squeeze(indexes, 1)

def CreateSmallSample():
    h5File = h5py.File("G:/PointCloud DataSets/semenatic3d/rawTrain/bildstein_station3_xyz_intensity_rgb.hdf5", 'r')
    pc = np.array(h5File["pointcloud"])
    lbls = np.array(h5File["labels"])
    indexes = np.random.choice(len(pc), 1000000, replace = False)

    h5File = h5py.File("G:/PointCloud DataSets/semenatic3d/rawTrain/small.hdf5", 'w')
    h5File.create_dataset("pointcloud", data=pc[indexes], dtype='float32', compression="lzf")
    h5File.create_dataset("labels", data=lbls[indexes], dtype='int8', compression="lzf")
    h5File.close()

def TestReadTimes():
    t = time()
    h5File = h5py.File("G:/PointCloud DataSets/semenatic3d/rawTrain/sg27_station2_intensity_rgb.hdf5", 'r')
    pc = h5File["pointcloud"]
    lbl = h5File["labels"]
    data = np.concatenate((pc, lbl), axis=1)
    h5File.close()
    print(data.shape)
    print("done reading hdf5 in {}:{} min.".format(int((time() - t)/60), int((time() - t)%60))) #09:04
    del pc
    del lbl
    del data
    pc = None
    lbl = None
    data = None

    t = time()
    data = np.load("G:/PointCloud DataSets/semenatic3d/rawTrain/sg27_station2_intensity_rgb.npy")
    print(data.shape)
    print("done reading npy in {}:{} min.".format(int((time() - t)/60), int((time() - t)%60))) #03:04

def VisualizePointCloudClassesAsync(pcFile, lblFile = None, downSample = True, noLbl = False, windowName = ""):
    if(len(windowName) == 0):
        if(not (lblFile is None)):
            windowName = Paths.FileName(lblFile)
        else:
            windowName = Paths.FileName(pcFile)

    p = Process(target=VisualizePointCloudClasses, args=(pcFile, lblFile, downSample, noLbl, windowName))
    p.start()

def VisualizePointCloudClasses(pcFile, lblFile = None, downSample = True, noLbl = False, windowName = "Pointcloud", noRgb = False, errorPoints = None, delPoints = None):
    xyz = None
    lbl = None
    rgb = None

    xyz = ReadXYZ(pcFile)

    if(not noRgb):
        rgb = ReadRGB(pcFile)
    if(not (lblFile is None) and not noLbl):
        lbl = ReadLabels(lblFile)

    SeparateClassesAndDisplay(xyz, rgb, lbl, downSample, windowName, errorPoints = errorPoints, delPoints = delPoints)

def VisualizeMultiplePointCloudLabels(ptsFile, lblFiles, downSample = False):
    for lbl in lblFiles:
        VisualizePointCloudClassesAsync(ptsFile, lbl, downSample=downSample)
    input1 = input()

def SeparateClassesAndDisplay(xyz, rgb = None, lbl = None, downSample = False, windowName = "Pointcloud", errorPoints = None, delPoints = None):
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
    # colors = [grey, red, mint, teal, blue, verygreen, brown, white, green]

    #NPM3D
    if(np.max(lbl) == 9):
        colors = [grey, red, blue, teal, mint, brown, pink, black, green]
    #Semantic3D
    elif(np.max(lbl) == 8):
        colors = [grey, verygreen, green, mint, red, blue, brown, black]
    elif(np.max(lbl) == 7):
        lbl += 1
        colors = [grey, verygreen, green, mint, red, blue, brown, black]
    
    errPts = None
    if(not (errorPoints is None)):
        if(len(errorPoints.shape) > 1):
            errorPoints = np.squeeze(errorPoints, 1)
            delPoints = np.squeeze(delPoints, 1)

        errPts = xyz[errorPoints]

        leavePts = (np.logical_not(errorPoints) != delPoints)
        if(not (delPoints is None)):
            xyz = xyz[leavePts]
        if(not (lbl is None)):
            lbl = lbl[leavePts]

    dataTool = DataTool()
    if not (lbl is None):
        xyz = np.concatenate((xyz, lbl), 1)
        classesPts = SeparateEachClass(xyz)
        colors = colors[:len(classesPts)]

        if(not (errPts is None)):
            classesPts.append(errPts)
            colors.append([1,0,0])

        dataTool.VisualizePointCloud(classesPts, colors, downSample=downSample, windowName=windowName)
    else:
        xyz = [xyz, errPts]
        rgb = [rgb, [1,0,0]]

        dataTool.VisualizePointCloud(xyz, rgb, downSample=downSample, windowName=windowName)

def WriteLabelsToLas(pcFile, lbl):
    if(not isinstance(lbl, np.ndarray)):
        lbl = np.array(pd.read_csv(lbl, dtype=np.int8, header=None))

    if(len(lbl.shape) != 1):
        lbl = np.squeeze(lbl, axis=-1)
    
    lbl = lbl.astype(np.uint8)

    lasFile = laspy.file.File(pcFile, mode = "r")
    lasFile.classification = lbl

    newFile = os.path.splitext(pcFile)[0]+"_labeled.las"
    lasFile.write_to_file(newFile)

def SaveSeparateClassLasFiles(pcFile, lblFile = None, saveFolder = None):

    xyz = None
    if(pcFile.endswith(".las")):
        lasFile = laspy.file.File(pcFile, mode = "r")
        xyz = np.concatenate((np.expand_dims(lasFile.x,1), np.expand_dims(lasFile.y,1), np.expand_dims(lasFile.z,1)), 1)
        if(not(lblFile is None)):
            lbl = np.array(pd.read_csv(lblFile, dtype=np.int8, header=None))
            xyz = np.concatenate((xyz, lbl), 1) 
        else:
            xyz = np.concatenate((xyz, np.expand_dims(lasFile.Classification, 1)), 1)
        lasFile.close()
    elif(pcFile.endswith(".hdf5")):
        h5File = h5py.File(pcFile, 'r')
        xyz = h5File["pointcloud"][:, :3]
        if("labels" in h5File and lblFile is None):
            lbl = np.array(h5File["labels"]).astype(np.int8)
            xyz = np.concatenate((xyz, lbl), 1)
        else:
            lbl = np.array(pd.read_csv(lblFile, dtype=np.int8, header=None))
            xyz = np.concatenate((xyz, lbl), 1)
        h5File.close()

    if(saveFolder is None):
        saveFolder = os.path.dirname(pcFile)
    nameBase = splitext(basename(pcFile))[0]

    classCount = int(np.max(xyz[:,3])+1)
    for i in range(classCount):
        pc = GetLabelPoint(xyz, i)
        if(len(pc[0])>0):
            ext = "(class_{}).las".format(i)
            lasFile = laspy.file.File(join(saveFolder, nameBase+ext), mode = "w", header=laspy.header.Header())        
            lasFile.X = pc[0][:,0]
            lasFile.Y = pc[0][:,1]
            lasFile.Z = pc[0][:,2]
            lasFile.close()

def VisualizeOnlyOneClass(classToMark, xyz, lbl = None, color = [1,0,0], downSample = True):
    if(not (lbl is None)):
        if(len(lbl.shape) == 1):
            lbl = np.expand_dims(lbl, 1)
        xyz = np.concatenate((xyz, lbl), 1)

    indices = np.where(xyz[:,3]==classToMark)
    classPoints = xyz[indices[0], :3]
    nonClassPoints = np.delete(xyz[:,:3], indices, 0)

    dataTool = DataTool()
    dataTool.VisualizePointCloud([nonClassPoints, classPoints], [None, color], downSample=downSample)

def VisualizeOneClassError(classToMark, xyz, lbl, trueLbl):
    indices = np.where(((lbl==trueLbl) & (trueLbl==classToMark)))[0]
    goodPoints = xyz[indices]
    xyz = np.delete(xyz, indices, 0)
    lbl = np.delete(lbl, indices, 0)
    trueLbl = np.delete(trueLbl, indices, 0)
    
    indices = np.where(((lbl!=trueLbl) & (lbl==classToMark)))[0]
    falsePositivePoints = xyz[indices]
    xyz = np.delete(xyz, indices, 0)
    lbl = np.delete(lbl, indices, 0)
    trueLbl = np.delete(trueLbl, indices, 0)

    indices = np.where(((lbl!=trueLbl) & (trueLbl==classToMark)))[0]
    falseNegativePoints = xyz[indices]
    xyz = np.delete(xyz, indices, 0)
    lbl = np.delete(lbl, indices, 0)
    trueLbl = np.delete(trueLbl, indices, 0)

    otherPoints = xyz

    white = np.array([255, 255, 255])/255
    red = np.array([230, 25, 75])/255
    green = np.array([60, 180, 75])/255

    dataTool = DataTool()
    dataTool.VisualizePointCloud([otherPoints, goodPoints, falsePositivePoints, falseNegativePoints], [None, white, red, green], downSample=False)

def VoxelizedDataOnTop(pcPath, voxelPath):
    xyz = ReadXYZ(pcPath)

    xyzrgbl = np.load(voxelPath)
    xyzrgbl = xyzrgbl[:,:3]

    dataTool = DataTool()
    dataTool.VisualizePointCloud([xyz, xyzrgbl], [None, [1,1,1]], downSample=False)

def ShowDemo():
    print("Show voxel centers")
    VoxelizedDataOnTop(Paths.rawTrain+"/bildstein_station3_xyz_intensity_rgb.hdf5", 
                        Paths.processedTrain+"/bildstein_station3_xyz_intensity_rgb.npy")

    # [manMadeTerrain, naturalTerrain, highVegetation, lowVegetation, buildings, hardScape, cars]
    # [grey,           teal,           green,           mint,         brown,      grey,      red]
    print("Show classes in color")
    VisualizePointCloudClasses(Paths.rawTrain+"/bildstein_station3_xyz_intensity_rgb.hdf5", Paths.generatedTest+"/bildstein_station3_xyz_intensity_rgb.labels")
    
    xyz = ReadXYZ(Paths.rawTrain+"/bildstein_station3_xyz_intensity_rgb.hdf5")
    lbl = ReadLabels(Paths.generatedTest+"/bildstein_station3_xyz_intensity_rgb.labels")
    print("Show cars in color")
    VisualizeOnlyOneClass(Label.cars, xyz, lbl, downSample=False)

    # [otherPoints, goodPoints, falsePositivePoints, falseNegativePoints]
    # [None,        white,      red,                 green]
    # keep in mind that this pointcloud has a lot of unlabeled points    
    trueLbl = ReadLabels(Paths.rawTrain+"/bildstein_station3_xyz_intensity_rgb.hdf5")
    print("Show car points error")
    VisualizeOneClassError(Label.cars, xyz, lbl, trueLbl)
    print("Show scanningArtefacts points error")
    VisualizeOneClassError(Label.scanningArtefacts, xyz, lbl, trueLbl)

    fileName = "30411934"
    xyzl = ReadXYZL("G:/PointCloud DataSets/"+fileName+".las", "G:/PointCloud DataSets/semenatic3d/generatedTest/"+fileName+".labels")
    print("Show dataset scanned by other party (classes in color)")
    VisualizePointCloudClasses("G:/PointCloud DataSets/30411934.las", "G:/PointCloud DataSets/semenatic3d/generatedTest/30411934.labels")

    fileName = "carquefou"
    xyzl = ReadXYZL("G:/PointCloud DataSets/"+fileName+".las", "G:/PointCloud DataSets/semenatic3d/generatedTest/"+fileName+".labels")
    print("Show another dataset scanned by other party (classes in color)")
    print("I have a feeling that this fails because of feet and meter factor")
    VisualizePointCloudClasses("G:/PointCloud DataSets/carquefou.las", "G:/PointCloud DataSets/semenatic3d/generatedTest/carquefou.labels")


    print("Show datasets from training test set (1)")
    VisualizePointCloudClasses(Paths.rawTest+"/castleblatten_station5_xyz_intensity_rgb.hdf5", Paths.generatedTest+"/castleblatten_station5_xyz_intensity_rgb.labels")
    print("Show datasets from training test set (2)")
    VisualizePointCloudClasses(Paths.rawTest+"/marketplacefeldkirch_station1_intensity_rgb.hdf5", Paths.generatedTest+"/marketplacefeldkirch_station1_intensity_rgb.labels")
    print("Show datasets from training test set (3)")
    VisualizePointCloudClasses(Paths.rawTest+"/stgallencathedral_station1_intensity_rgb.hdf5", Paths.generatedTest+"/stgallencathedral_station1_intensity_rgb.labels")
    print("Show datasets from training test set (4)")
    VisualizePointCloudClasses(Paths.rawTest+"/birdfountain_station1_xyz_intensity_rgb.hdf5", Paths.generatedTest+"/birdfountain_station1_xyz_intensity_rgb.labels")
    print("Show datasets from training test set (5)")
    VisualizePointCloudClasses(Paths.rawTest+"/sg27_station10_intensity_rgb.hdf5", Paths.generatedTest+"/sg27_station10_intensity_rgb.labels")

def SplitDataFiles(srcFolder, dstFolder, fileCount):
    assert fileCount >= 1

    if(not os.path.isdir(dstFolder)):
        os.mkdir(dstFolder)
    
    files = np.array(Paths.GetFiles(srcFolder))
    indexes = np.random.choice(len(files), int(fileCount), replace = False)
    filesToMove = files[indexes]

    for file in filesToMove:
        newName = os.path.join(dstFolder, basename(file))
        os.rename(file, newName)

def Part(a, b):
    return "{:.1f}%".format(len(a)/len(b)*100)

def PartInt(a, b):
    return "{:.1f}%".format(a/b*100)

def ConvertData():
    file = r"G:\PointCloud DataSets\semantic3d\rawTrain\bildstein_station3_xyz_intensity_rgb.hdf5"

    h5File = h5py.File(file, 'r')
    pc = h5File["pointcloud"]
    h5File.close()

def DownSampleVGTUData():
    # path = r"G:\PointCloud DataSets\VGTU_2020.03.19.las"
    path = r"G:\PointCloud DataSets\VGTU\VGTU_limitbox.las"

    xyz, rgb = ReadXYZRGB(path)

    print("Processing...")
    pc = Voxelize(xyz, rgb, None)
    print("pc.shape: ", pc.shape)
    print("Saving...")
    np.save(r"G:\PointCloud DataSets\VGTU\VGTU_limitbox_voxels.npy", pc, allow_pickle=False)
    print("Done.")

def LiveSegmentationAnimation(ptsFile, lblFile = None):

    pts = ReadXYZ(ptsFile)

    if(lblFile is None):
        rgb = ReadRGB(ptsFile)
    else:
        lbl = ReadLabels(lblFile)
        delIdx = np.where(lbl == 0)[0]
        pts = np.delete(pts, delIdx, 0)
        lbl = np.delete(lbl, delIdx, 0)
        rgb = np.zeros((len(lbl), 3), np.float)
        for i in range(0, len(Colors.Sema3D)):
            indexes = np.where(lbl == i+1)[0]
            rgb[indexes] = Colors.Sema3D[i]
    
    # DataTool().VisualizePointCloud([pts], [rgb], animationFunction=AnimationCallBack, downSample=False)
    # DataTool().VisualizePointCloud([pts], [rgb], animationFunction=PlayTrajectory().StepTrajectory, downSample=False)
    DataTool().VisualizePointCloud([pts], [rgb], animationFunction=AnimationCallBack, downSample=False, recordTrajectory = True)

def MeasureIOU():
    from ConvPoint import UpscaleToOriginal
    from sklearn.metrics import confusion_matrix
    from metrics import stats_accuracy_per_class, stats_iou_per_class

    org_pts = ReadXYZ(r"G:\PointCloud DataSets\NPM3D\torch_generated_data\train_pointclouds\Lille1_1_0.npy")
    src_pts = ReadXYZ(r"G:\PointCloud DataSets\NPM3D\processedTrainVoxels\Lille1_1_0.npy")
    

    org_lbl = ReadLabels(r"G:\PointCloud DataSets\NPM3D\torch_generated_data\train_pointclouds\Lille1_1_0.npy")
    src_lbl = ReadLabels(r"G:\PointCloud DataSets\NPM3D\processedTrainVoxels\Lille1_1_0.npy")
    
    org_lbl = np.squeeze(org_lbl, 1)    
    src_lbl = np.squeeze(src_lbl, 1)

    new_lbl = UpscaleToOriginal(org_pts, src_pts, src_lbl)

    cm = confusion_matrix(org_lbl, new_lbl)
    avg_acc, _ = stats_accuracy_per_class(cm)
    avg_iou, _ = stats_iou_per_class(cm)

def VisualizeError():
    xyz = ReadXYZ(r"G:\PointCloud DataSets\NPM3D\torch_generated_data\train_pointclouds\Lille1_1_0.npy")
    org_lbl = ReadLabels(r"G:\PointCloud DataSets\NPM3D\torch_generated_data\train_pointclouds\Lille1_1_0.npy")
    new_lbl = ReadLabels(r"G:\PointCloud DataSets\NPM3D\generatedTest\Lille1_1_0_NPM3D(NOCOL)_39_train(91.4)_val(87.8).txt")

    delPts = np.where(org_lbl == 0)[0]        
    xyz = np.delete(xyz, delPts, 0)
    org_lbl = np.delete(org_lbl, delPts, 0)
    new_lbl = np.delete(new_lbl, delPts, 0)

    errorIdx = np.where(org_lbl != new_lbl)[0]
    errorPts = xyz[errorIdx]
    xyz = np.delete(xyz, errorIdx, 0)
    org_lbl = np.delete(org_lbl, errorIdx, 0)

    pts_colors = np.zeros((len(xyz), 3), np.float)
    for i in range(0, len(Colors.Npm3D)):
        indexes = np.where(org_lbl == i+1)[0]
        pts_colors[indexes] = Colors.Npm3D[i]

    DataTool().VisualizePointCloud([xyz, errorPts], [pts_colors, [1,0,0]], downSample=False)

def PostProcessCurbScores(scores):
    pred = np.zeros((scores.shape[0],))
    
    truePoints = np.where(scores.argmax(1) == 1)[0]
    pred[truePoints] = 1

    possiblePoints = np.unique(np.concatenate([truePoints, np.where(scores[:, 1] > 50)[0]]))
    pred[possiblePoints] = 1

    # pred[np.where(np.abs(scores[:, 0] - scores[:, 1]) > 5)[0]] = 1

    return pred

def SplitCurbs(curbPts, distance = 3):
    import scipy.cluster.hierarchy as hcluster    

    clusters = hcluster.fclusterdata(curbPts, distance, criterion="distance")
    clusters = [curbPts[np.where(clusters == i)[0]] for i in set(clusters)]

    return clusters

def RandomColor():    
    return np.random.uniform(0,1, [3,])

def CutCurb(pts):
    from matplotlib import pyplot as plt

    N = 300
    t = np.linspace(0, 2*np.pi, N)
    x = 5*np.cos(t) + 0.2*np.random.normal(size=N) + 1
    y = 4*np.sin(t+0.5) + 0.2*np.random.normal(size=N)
    plt.plot(x, y, '.')     # given points
    # plt.show()

    xmean, ymean = x.mean(), y.mean()
    x -= xmean
    y -= ymean
    U, S, V = np.linalg.svd(np.stack((x, y)))

    tt = np.linspace(0, 2*np.pi, 1000)
    circle = np.stack((np.cos(tt), np.sin(tt)))    # unit circle
    transform = np.sqrt(2/N) * U.dot(np.diag(S))   # transformation matrix
    fit = transform.dot(circle) + np.array([[xmean], [ymean]])
    plt.plot(fit[0, :], fit[1, :], 'r')
    plt.show()

    
    fitter = np.poly1d(np.polyfit(pts[:,1], pts[:,0], 2))
    curbY = np.linspace(np.min(pts[:,1]), np.max(pts[:,1]), 1000)
    curbX = fitter(curbY)

    newPts = np.array([[curbX[i], curbY[i], pts[:,2].max()] for i in range(len(curbX))])
    
    # numpy.polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False)[source]    
    # DataTool().VisualizePointCloud([pts, newPts], dataColors = [[0,0,1], [1,0,0]])

    return newPts

def PointsInBlock(xyz, pt, blockSize):
    mask_x = np.logical_and(xyz[:,0]<pt[0]+blockSize/2, xyz[:,0]>pt[0]-blockSize/2)
    mask_y = np.logical_and(xyz[:,1]<pt[1]+blockSize/2, xyz[:,1]>pt[1]-blockSize/2)
    mask = np.logical_and(mask_x, mask_y)
    return xyz[mask]

def ChangeLabel(original_array, destination_array, target_label, new_label):
    idx = np.where(original_array == target_label)[0]
    destination_array[idx] = new_label
    return destination_array

def ChangeLabels(labels, label_map : dict):
    destination_array = np.copy(labels)
    for target_label, new_label in label_map.items():
        destination_array = ChangeLabel(labels, destination_array, target_label, new_label)
    return destination_array

def RemoveClasses(lbl, remove_labels : list, pts, ftr = None):
    
    mask = np.invert(np.isin(lbl, remove_labels))
    mask = np.where(mask)[0]
    
    lbl = lbl[mask]
    
    if not(pts is None):
        pts = pts[mask]    
    if not(ftr is None):
        ftr = ftr[mask]
    
    return pts, ftr, lbl

def PreprocessSemantic3DExtended():
    label_map = {
        1 : 0, # unclassified -> unlabeled points
        2 : 1, # ground -> "natural terrain"
        5 : 3, # high veg -> high veg
        3 : 4, # low veg -> low veg        
        6 : 5, # building -> building
        19 : 6, # hard scape
        7 : 7, # low point -> artefacts
        20 : 8, # cars
        
        15 : 6, # poles and signs -> hard scape
        16 : 6, # wires -> hard scape     
    }
    
    PreprocessDatasetToVoxels(r"D:\semantic3d\rawTrainManual", Paths.Semantic3D.processedTrain, extension = ".las", saveFormat=".las", override=False, label_map = label_map)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

from multiprocessing import Pool

def SplitFilesIntoChunks(input_folder, ext, consts, output_folder, save_ext, summary_file, blockSize, step = None, multiProcess = True):

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    
    files = Paths.GetFiles(input_folder, findExtesions=[ext])
    print(f"Files to split: {len(files)}")    
            
    split_func = partial(SplitPointCloudsIntoChunks, output_folder = output_folder, blockSize = blockSize, step = step, save_ext = save_ext, NormalizeDataFunc = consts.NormalizeData)
    
    if(not multiProcess):
        summary_lists = [split_func(file) for file in tqdm(files)]
    else:    
        try:    
            with Pool(8) as p:
                summary_lists = list(tqdm(p.imap(split_func, files), total=len(files)))
        except Exception as e:
            print(str(e))
    
    summary = { 'File':[], "Class" : [], "NumPoints" : [] }
    
    for file_name, point_count, class_points in tqdm(np.concatenate(summary_lists)):
        for cls_number, count in class_points:
            summary['File'].append(file_name)
            summary['Class'].append(cls_number)
            summary['NumPoints'].append(count)
        
    pd.DataFrame.from_dict(summary).to_csv(summary_file, index=False)

def SplitPointCloudsIntoChunks(file, output_folder, blockSize, step = None, save_ext = '.npy', remove_classes = None, NormalizeDataFunc = None):
            
    file_name = os.path.splitext(os.path.basename(file))[0]
    data = ReadXYZRGBLBL(file)
    xyz, ftr, lbl = np.split(data, [3, data.shape[1]-1], axis=-1)
    if(ftr.shape[1] == 0):
        ftr = None
    if(not NormalizeDataFunc is None):
        xyz, ftr, lbl = NormalizeDataFunc(xyz, ftr, lbl, validation = False)
    
    chunkSize = sqrt(blockSize*blockSize*2) # make each chunk width equal to block hypotenuse
    if(step is None):
        step = chunkSize*0.7
    discretized = ((xyz[:,:2]).astype(float)/step).astype(int)
    chunk_centers = np.unique(discretized, axis=0).astype(np.float32)*step + step/2
        
    # boxes = [BoundingBoxFromVoxel([pt[0], pt[1], xyz[:,2].min()], [chunkSize, chunkSize, 10]) for pt in chunk_centers]
    # colors = [np.random.uniform(0, 1, 3) for _ in range(len(boxes))]
    # DataTool().VisualizePointCloudAsync(xyz, bBoxes = boxes, boxesColors = colors, windowName=f"Chunks count: {len(boxes)}")
    
    summary = []
    
    for i, center in enumerate(chunk_centers):
        mask_x = np.logical_and(xyz[:,0]<center[0]+chunkSize/2, xyz[:,0]>center[0]-chunkSize/2)
        mask_y = np.logical_and(xyz[:,1]<center[1]+chunkSize/2, xyz[:,1]>center[1]-chunkSize/2)
        mask = np.where(np.logical_and(mask_x, mask_y))[0]
        
        if(len(mask) < 1000):
            continue
        
        save_xyz = xyz[mask]
        save_xyz = save_xyz - np.array([center[0], center[1], save_xyz[:,2].min()]) # place each chunk center at [0,0,0]
        save_fts = None if (ftr is None) else ftr[mask]
        save_lbl = lbl[mask]
        
        # if(i > 500):
        #     DataTool().VisualizePointCloudAsync(save_xyz, bBoxes=[BoundingBoxFromVoxel([0, 0, 0], [blockSize, blockSize, 20])], windowName="Original")
        #     DataTool().VisualizePointCloudAsync(RotatePointCloud(save_xyz), bBoxes=[BoundingBoxFromVoxel([0, 0, 0], [blockSize, blockSize, 20])], windowName="With rotation")

        output_file_name = file_name+f"_{i}"
        output_file = os.path.join(output_folder, output_file_name+save_ext)
        # output_file = os.path.join(output_folder, output_file_name+".las")
        # save_data = np.concatenate([save_xyz, save_lbl], axis=1) if (save_fts is None) else np.concatenate([save_xyz, save_fts, save_lbl], axis=1)
        # np.save(output_file, save_data, allow_pickle=False)
        SavePointCloud(output_file, save_xyz, intensity = save_fts, labels = save_lbl)
        
        summary.append((output_file_name, len(save_xyz), np.stack(np.unique(save_lbl, return_counts=True),axis=1)))
    
    return summary

if __name__ == "__main__":
    # print()    
    # xyz = ReadXYZ("E:/ParisData/preprocessedTrain/Joinville.las")
    # lbl = ReadLabels("E:/ParisData/preprocessedTrain/Joinville.las")
    # SaveToLas("E:/ParisData/preprocessedTrain/Joinville_xyz_only.las", xyz, labels = lbl)
    
    files = Paths.GetFiles(Paths.Paris.processedTrain)
    clouds = [ReadLabels(file) for file in files]
    labels = np.concatenate(clouds)
    unique = np.unique(labels)
    
    PreprocessDatasetToVoxels(Paths.Paris.rawTrain, Paths.Paris.processedTrain, extension = ".las", saveFormat=".las", override=False)

    # file = "/media/jonas/SSD Extreme/semantic3d/processedTest/MM_Boulder.npy"
    # predFile = "C:/git_repos/ConvPoint_Keras/data/Curbs(7_1)(noFeature)(Rotate)_21bdbe6aa82d4e259526ab46577e795a_25_train(75.1)_val(60.7)/park_extracted.labels.npy"
    # curbScores = "C:/git_repos/ConvPoint_Keras/data/Curbs(7_1)(noFeature)(Rotate)_21bdbe6aa82d4e259526ab46577e795a_25_train(75.1)_val(60.7)/park_extracted_scores.npy"

    # file = "C:\git_repos\LineExtraction\data\preprocessed_data/SmallArea2.npy"
    # predFile = "C:/git_repos/ConvPoint_Keras/data/Curbs(7_1)(noFeature)(Rotate)_21bdbe6aa82d4e259526ab46577e795a_25_train(75.1)_val(60.7)/SmallArea2.labels.npy"
    # curbScores = "C:/git_repos/ConvPoint_Keras/data/Curbs(7_1)(noFeature)(Rotate)_21bdbe6aa82d4e259526ab46577e795a_25_train(75.1)_val(60.7)/SmallArea2_scores.npy"

    # xyz = ReadXYZ(file)
    # rgb = ReadRGB(file)
    # DataTool().VisualizePointCloudAsync([xyz], [rgb])
    # true = ReadLabels(file)
    
    # pred1 = np.squeeze(ReadLabels(predFile), axis=1)
    # linePoints = np.where(pred1 == 1)[0]
    # DataTool().VisualizePointCloudAsync([np.delete(xyz, linePoints, axis=0), xyz[linePoints]], [np.delete(rgb, linePoints, axis=0), [1,0,0]], windowName="pred1")
        
    # pred2 = PostProcessCurbScores(np.load(curbScores))
    # linePoints = np.where(pred2 == 1)[0]
    # DataTool().VisualizePointCloudAsync([np.delete(xyz, linePoints, axis=0), xyz[linePoints]], [np.delete(rgb, linePoints, axis=0), [1,0,0]], windowName="pred2")

    # pts = xyz[linePoints]
    # DataTool().VisualizePointCloudAsync([pts])
    # curbs = SplitCurbs(pts, distance = 5)
    # curbsColors = [RandomColor() for i in range(len(curbs))]
    # DataTool().VisualizePointCloudAsync(curbs, dataColors = curbsColors, windowName="Clusters")

    # linePoints = [CutCurb(curb) for curb in curbs if len(curb) > 100]
    # lineColors = [[1,0,0] for i in range(len(linePoints))]
    # DataTool().VisualizePointCloudAsync(curbs + linePoints, dataColors = curbsColors + lineColors)

    # DataTool().VisualizePointCloudAsync([xyz] + linePoints, [rgb] + lineColors, windowName="Result")


    # linePoints = np.where(true == 1)[0]
    # DataTool().VisualizePointCloud([np.delete(xyz, linePoints, axis=0), xyz[linePoints]], [np.delete(rgb, linePoints, axis=0), [1,0,0]], windowName="Test data")

    # ConvertToLas("G:/PointCloud DataSets/semantic3d/rawTrain", "G:/PointCloud DataSets/semantic3d/rawTrainLas")

    # mainPC = ReadRGB(r"G:\PointCloud DataSets\NPM3D\torch_generated_data\train_pointclouds\Lille2_0.npy")

    # mainPC = ReadXYZ(r"G:\PointCloud DataSets\NPM3D\training_10_classes\Lille2.ply")
    # train = [r"G:\PointCloud DataSets\NPM3D\torch_generated_data\train_pointclouds\Lille2_0.npy",
    #         r"G:\PointCloud DataSets\NPM3D\torch_generated_data\train_pointclouds\Lille2_1.npy",
    #         r"G:\PointCloud DataSets\NPM3D\torch_generated_data\train_pointclouds\Lille2_2.npy",            
    #         r"G:\PointCloud DataSets\NPM3D\torch_generated_data\train_pointclouds\Lille2_8.npy",
    #         r"G:\PointCloud DataSets\NPM3D\torch_generated_data\train_pointclouds\Lille2_9.npy"]
    # test = r"G:\PointCloud DataSets\NPM3D\torch_generated_data\train_pointclouds\Lille2_10.npy"
    # nextTrain = r"G:\PointCloud DataSets\NPM3D\torch_generated_data\train_pointclouds\Lille2_3.npy"

    # goodPoints = []
    # colors = []
    # for file in train:
    #     goodPoints.append(ReadXYZ(file))
    #     colors.append(np.array([0.0, 1.0, 0.0]))

    # goodPoints.append(ReadXYZ(test))
    # colors.append(np.array([1.0, 0.0, 0.0]))

    # goodPoints.append(ReadXYZ(nextTrain))
    # colors.append(np.array([0.0, 0.0, 1.0]))

    # goodPoints.append(mainPC)
    # colors.append(None)
    

    # DataTool().VisualizePointCloud(goodPoints, colors, downSample=True)


    # file = r"G:\PointCloud DataSets\semantic3d\processedTrain\domfountain_station3_xyz_intensity_rgb_voxels.npy"
    # file = r"G:\PointCloud DataSets\VGTU\VGTU_Clean_voxels.npy"
    # lbl = None
    # # lbl = r"G:\PointCloud DataSets\VGTU\VGTU_Clean_Sem3D(fusion)_1_train(88.9)_val(79.7).labels"
    # LiveSegmentationAnimation(file, lbl)

    # DownSampleVGTUData()
    # ConvertData()
    # VisualizePointCloudClasses(r"G:\PointCloud DataSets\NPM3D\torch_generated_data\train_pointclouds\Lille1_1_0.npy", 
    #                             r"G:\PointCloud DataSets\NPM3D\generatedTest\Lille1_1_0_NPM3D(NOCOL)_39_train(91.4)_val(87.8).txt", downSample=False)
    # VisualizePointCloudClasses(r"G:\PointCloud DataSets\VGTU\VGTU_Clean.npy", r"G:\PointCloud DataSets\VGTU\VGTU_Clean.labels", downSample=True)
    # VisualizePointCloudClasses(r"C:\Users\Jonas\Downloads\bildstein_station1_xyz_intensity_rgb_voxels.npy", noLbl = True, downSample=False)
    
    # VisualizePointCloudClassesAsync("G:/PointCloud DataSets/NPM3D/test_10_classes/ajaccio_57.ply", 
    #                                 "G:/PointCloud DataSets/NPM3D/generatedTest/NPM3D(NOCOL)(FullAugment)_49_train(89.6)_val(89.2)/ajaccio_57.txt", downSample=False, windowName="NPM3D(NOCOL)(FullAugment)_49")

    # VisualizePointCloudClassesAsync("G:/PointCloud DataSets/NPM3D/test_10_classes/ajaccio_57.ply", 
    #                                 "G:/PointCloud DataSets/NPM3D/generatedTest/generated_NPM3D(NOCOL)_39_train(91.4)_val(87.8)/ajaccio_57.txt", downSample=False, windowName="NPM3D(NOCOL)_39")

    # VisualizePointCloudClassesAsync("G:/PointCloud DataSets/NPM3D/test_10_classes/ajaccio_57.ply", "G:/PointCloud DataSets/NPM3D/processedTest/ajaccio_57.txt", downSample=False, windowName="keras")
    # VisualizePointCloudClassesAsync("G:/PointCloud DataSets/NPM3D/test_10_classes/ajaccio_57.ply", "G:/PointCloud DataSets/NPM3D/torch_generated_data/results88.2%/ajaccio_57.txt", downSample=False , windowName="torch")
    # VisualizePointCloudClassesAsync("G:/PointCloud DataSets/NPM3D/test_10_classes/ajaccio_57.ply", "G:/PointCloud DataSets/NPM3D/processedTest/ajaccio_57.txt", downSample=False)
    # VisualizePointCloudClassesAsync("G:/PointCloud DataSets/NPM3D/test_10_classes/dijon_9.ply", "G:/PointCloud DataSets/NPM3D/processedTest/dijon_9.txt", downSample=False)
    # VisualizePointCloudClassesAsync("G:/PointCloud DataSets/NPM3D/test_10_classes/ajaccio_2.ply", "G:/PointCloud DataSets/NPM3D/5nd_submission/results/ajaccio_2.txt", downSample=False, windowName="second")
    # VisualizePointCloudClasses("G:/PointCloud DataSets/NPM3D/training_10_classes/Paris.ply", downSample = False)
    # VisualizePointCloudClasses("G:/PointCloud DataSets/NPM3D/training_10_classes/Lille2.ply", downSample = False)
    # VisualizePointCloudClassesAsync("G:/PointCloud DataSets/NPM3D/processedTrain/Lille1_1_0.npy")
    # VisualizePointCloudClassesAsync("G:/PointCloud DataSets/NPM3D/training_10_classes/Lille1_1.ply", noLbl = True)


    # VisualizePointCloudClassesAsync("G:/PointCloud DataSets/NPM3D/test_10_classes/dijon_9.ply", r"C:\Users\Jonas\Downloads\dijon_9_NPM3D(NOCOL)_26_89.5.txt", downSample=False)
    # VisualizePointCloudClassesAsync("G:/PointCloud DataSets/NPM3D/test_10_classes/dijon_9.ply", r"C:\Users\Jonas\Downloads\dijon_9_NPM3D(NOCOL)_30_90.9.txt", downSample=False)
    # input1 = input()
    
    # ptsFile = "G:/PointCloud DataSets/semantic3d/rawTest/birdfountain_station1_xyz_intensity_rgb.hdf5"
    # lblFiles = [
    #             # r"C:\Users\Jonas\Downloads\birdfountain_station1_xyz_intensity_rgb_Sem3D(RGB)_41_87.1.labels",
    #             # r"C:\Users\Jonas\Downloads\birdfountain_station1_xyz_intensity_rgb_Sem3D(RGB)_22_82.7.labels",
    #             # r"C:\Users\Jonas\Downloads\birdfountain_station1_xyz_intensity_rgb_Sem3D(RGB)_29_84.4.labels",
    #             # r"C:\Users\Jonas\Downloads\birdfountain_station1_xyz_intensity_rgb_Sem3D(RGB)_41_86.7.labels",
    #             r"G:\PointCloud DataSets\semantic3d\generatedTest\birdfountain1.labels",
    #             ]
    # VisualizeMultiplePointCloudLabels(ptsFile, lblFiles, downSample = False)

    # ptsFile = r"G:\PointCloud DataSets\NPM3D\test_10_classes\dijon_9.ply"
    # lblFiles = [
    #             r"C:\Users\Jonas\Downloads\dijon_9_NPM3D(RGB)_21_train(86.1)_val(86.8).txt",
    #             r"C:\Users\Jonas\Downloads\dijon_9_NPM3D(RGB)_26_train(87.6)_val(88.3).txt",
    #             r"C:\Users\Jonas\Downloads\dijon_9_NPM3D(RGB)_32_train(88.7)_val(89.3).txt",
    #             ]
    # VisualizeMultiplePointCloudLabels(ptsFile, lblFiles, downSample = False)

    # VisualizePointCloudClasses("G:/PointCloud DataSets/semantic3d/rawTest/birdfountain_station1_xyz_intensity_rgb.npy", 
    #                                 "G:/PointCloud DataSets/semantic3d/generatedTest/birdfountain_station1_xyz_intensity_rgb.labels", downSample=False, windowName="keras")

    # VisualizePointCloudClasses("G:/PointCloud DataSets/semantic3d/rawTest/birdfountain_station1_xyz_intensity_rgb.npy", 
    #                             r"G:\PointCloud DataSets\semantic3d\generatedTest\birdfountain1.labels", downSample=False, windowName="keras")

    # SplitDataFiles(Paths.NPM3D.processedTrain, Paths.NPM3D.processedTest, 3)

    # PreprocessDatasetToVoxels(Paths.Semantic3D.rawTestReduced, Paths.Semantic3D.processedTestReduced, override=False)
    # PreprocessDatasetToVoxels(Paths.Semantic3D.rawTrain, r"G:\PointCloud DataSets\semantic3d\processedTrain(0.15m)", override=False, extension=".hdf5", voxelSize=0.15)
    # PreprocessDatasetToVoxels(Paths.Semantic3D.rawTest, Paths.Semantic3D.processedTest, override=False, extension=".hdf5")
    # PreprocessDatasetToVoxels(Paths.Aerial.rawTrain, Paths.Aerial.processedTrain, extension='.laz')
    # PreprocessDatasetToVoxels(r"G:\PointCloud DataSets\NPM3D\\training_10_classes", Paths.NPM3D.processedTrain,  override=False, extension=".ply")
    # PreprocessDatasetToVoxels(Paths.rawTrain, "G:/PointCloud DataSets/semenatic3d/processedTrain(0.02_voxels)", override=False, voxelSize=0.02)

    # dataTool = DataTool()
    # dataTool.ConvertDatasets(Paths.Semantic3D.rawTestReduced, Paths.Semantic3D.rawTestReduced)
    # dataTool.ConvertToBin("G:/PointCloud DataSets/semenatic3d/train_txt/neugasse_station1_xyz_intensity_rgb.txt",
    #                       "G:/PointCloud DataSets/semenatic3d/train_txt/neugasse_station1_xyz_intensity_rgb.labels",
    #                       "G:/PointCloud DataSets/semenatic3d/rawTrain/neugasse_station1_xyz_intensity_rgb")#,".npy")

    # VisualizePointCloudClasses("G:/PointCloud DataSets/big_feet.las", Paths.generatedTest+"/big_feet.labels")
    # VisualizePointCloudClasses("G:/PointCloud DataSets/big_feet.las", noLbl=True)
    
    # WriteLabelsToLas("G:/PointCloud DataSets/monorail_bridge.las", "G:/PointCloud DataSets/semenatic3d/generatedTest/monorail_bridge.labels")
    # SaveSeparateClassLasFiles("G:/PointCloud DataSets/monorail_bridge.las")
    # SaveSeparateClassLasFiles("G:/PointCloud DataSets/semenatic3d/rawTrain/bildstein_station3_xyz_intensity_rgb.hdf5")

    # input()