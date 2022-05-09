from tensorflow.keras.utils import Sequence
import tensorflow as tf
from imports import np
from PIL import Image, ImageEnhance, ImageOps
from dataTool import *
from configs import Config
from configs.SDE import SDE
from configs.NMG import NMG
from voxelization import voxelize

class TrainSequence(Sequence):
    def __init__(self, iteration_number, consts : Config, dataAugmentation = True, randomPoints = True, balanceClasses = True, validation = False):
        self.cts = consts
        self.dataAugmentation = dataAugmentation
        self.iterations = iteration_number
        self.selectRandomPoints = randomPoints
        self.balanceClasses = balanceClasses
        self.validation = validation

    def __len__(self):
        return int(self.iterations)
    
    def on_epoch_end(self):
        if(not self.validation):
            self.cts.RotateClouds(self.validation)        
    
    def GenerateSampleFromFile(self, sample_idx : int, samples : int, oneClass):            
        # load the data
        # if sample_idx == -1, read files that contains all classes present
        # self.cts.GetClassFile(self.cts.GetData(sample_idx % self.cts.classCount if sample_idx != -1 else sample_idx, self.validation, self.mapLabels))
        pts, fts, lbs = self.cts.GetData(self.validation)                    
                        
        if(samples > 1):
            randomCenters = GetSampleCenters(pts, lbs, samples, self.balanceClasses, self.cts.blocksize, oneClass)
            
            if self.cts.input_tile_count == 1:
                dims = [samples, self.cts.npoints]
            elif self.cts.input_tile_count > 1:
                dims = [samples, self.cts.input_tile_count, self.cts.npoints]
                
            if not self.cts.noFeature:
                ftsList = np.zeros(dims + [self.cts.featureComponents], np.float32)
            else:
                ftsList = None
            ptsList = np.zeros(dims + [self.cts.pointComponents], np.float32)
            lbsList = np.zeros(dims + [self.cts.classCount], np.float32)
        
            for i, center in enumerate(randomCenters):
                temppts, tempfts, templbs = ProcessSample(pts, fts, lbs, center, self.cts.classCount, self.cts.blocksize, self.cts.npoints, self.selectRandomPoints)
                temppts, tempfts = AugmentSample(temppts, tempfts, self.cts.Mirror, self.cts.Rotate, self.cts.Scale, self.cts.Jitter, self.cts.FtrAugment)

                if not self.cts.noFeature:
                    ftsList[i] = np.expand_dims(tempfts, 0)
                ptsList[i] = np.expand_dims(temppts, 0)
                lbsList[i] = np.expand_dims(templbs, 0)
                
                return ptsList, ftsList, lbsList
        else:
            center = GetSampleCenters(pts, lbs, samples, self.balanceClasses, self.cts.blocksize)[0]
            temppts, tempfts, templbs = ProcessSample(pts, fts, lbs, center, self.cts.classCount, self.cts.blocksize, self.cts.npoints, self.selectRandomPoints)
            temppts, tempfts = AugmentSample(temppts, tempfts, self.cts.Mirror, self.cts.Rotate, self.cts.Scale, self.cts.Jitter, self.cts.FtrAugment)
            # DataTool().VisualizePointCloudAsync([temppts], classes = [templbs])
            return temppts, tempfts, templbs
      
    def __getitem__(self, sample_idx, batchSize = None, oneClass = None):
        batchSize = self.cts.BatchSize() if (batchSize is None) else batchSize
        
        if self.cts.oneSamplePerFile:
            ptsList = np.zeros([batchSize, self.cts.npoints, self.cts.pointComponents], dtype=np.float32)            
            lbsList = np.zeros([batchSize, self.cts.npoints, self.cts.classCount], dtype=np.float32)
            if(not self.cts.noFeature):
                ftsList = np.zeros([batchSize, self.cts.npoints, self.cts.featureComponents], dtype=np.float32)
            for i in range(batchSize):
                pts, fts, lbs = self.GenerateSampleFromFile(sample_idx, 1, oneClass)
                ptsList[i] = pts
                if(not fts is None):
                    ftsList[i] = fts
                lbsList[i] = lbs
        else:
            ptsList, ftsList, lbsList = self.GenerateSampleFromFile(sample_idx, batchSize, oneClass)
            
        if self.cts.noFeature:
            return ptsList, lbsList
        else: # works for RGB and fusion models
            return [ftsList, ptsList], lbsList

class TestSequence(Sequence):
    def __init__(self, filename, consts : Config, splitDataSetToParts = -1, localMachineCap = False, test = False, generateALl = False):
        self.filename = filename
        self.cts = consts
        self.batchSize = consts.BatchSize()
        self.npoints = consts.npoints
        self.nocolor = consts.noFeature
        self.featureComponents = consts.featureComponents
        self.fusion = consts.Fusion
        self.test = test
        self.generateALl = generateALl
        self.lbl = []

        pointcloud = voxelize(filename, self.cts.downsampling_voxel_size)

        lbl = pointcloud.labels if pointcloud.labels is not None else np.zeros((len(pointcloud.xyz), 1))
        data = np.concatenate([pointcloud.xyz, lbl], axis=1)
        xyz, rgb, self.lbl = np.split(data, [consts.pointComponents, data.shape[1]-1], axis=-1)
        # fix labels and filter unlabeled points
        # DataTool().VisualizePointCloud([xyz], classes = [self.lbl])
        xyz, rgb, self.lbl = consts.NormalizeData(xyz, rgb, self.lbl, validation = True)
        # DataTool().VisualizePointCloudAsync([xyz], classes = [self.lbl])
        if not(rgb is None):
            self.xyzrgblbl = np.concatenate((xyz, rgb), 1)
        else:
            self.xyzrgblbl = xyz
            
        if(len(self.xyzrgblbl) == len(xyz)):
            self.xyzrgblbl = np.concatenate((self.xyzrgblbl, self.lbl), 1)
        
        assert consts.test_step < 1, "test_step is percengate of blocksize, can't be greater than 1"
        
        step = consts.blocksize * consts.test_step
        discretized = ((self.xyzrgblbl[:,:2]).astype(float)/step).astype(int)
        self.allpts = np.unique(discretized, axis=0)
        self.allpts = self.allpts.astype(np.float32)*step
        # print("Test_step:", consts.test_step)
        # print("Step count:", len(self.allpts))

        if(consts.IsLocalMachine() and localMachineCap):
            self.allpts = self.allpts[:115] #small sample for testing

        self.splitDataSetToParts = splitDataSetToParts
        if(self.splitDataSetToParts != -1):
            self.ptIndex = 0
        else:
            self.pts = self.allpts
            self.idxList = np.zeros([len(self.pts), self.cts.input_tile_count, self.cts.npoints] if (self.cts.input_tile_count > 1) else [len(self.pts), self.cts.npoints], np.int64)

    def LenParts(self):
        if(self.splitDataSetToParts != -1):
            return ceil(len(self.allpts)/self.splitDataSetToParts)
        else:
            return 1

    def Reset(self):
        self.ptIndex = 0

    def NextPart(self):
        if(self.splitDataSetToParts <= 0):
            return False
        if(self.ptIndex >= len(self.allpts)):
            return False

        self.nextIndex = np.min([self.ptIndex+self.splitDataSetToParts, len(self.allpts)])
        self.pts = self.allpts[self.ptIndex : self.nextIndex]
        self.ptIndex = self.nextIndex

        self.idxList = np.zeros([len(self.pts), self.cts.input_tile_count, self.cts.npoints] if (self.cts.input_tile_count > 1) else [len(self.pts), self.cts.npoints], np.int64)
        return True

    def __len__(self):
        return ceil(len(self.pts)/self.batchSize)

    def compute_mask(self, pt, bs, ptsftslbl = None):
        if(ptsftslbl is None):
            ptsftslbl = self.xyzrgblbl
            
        # build the mask
        mask_x = np.logical_and(ptsftslbl[:,0]<pt[0]+bs/2, ptsftslbl[:,0]>pt[0]-bs/2)
        mask_y = np.logical_and(ptsftslbl[:,1]<pt[1]+bs/2, ptsftslbl[:,1]>pt[1]-bs/2)
        mask = np.logical_and(mask_x, mask_y)
                
        return np.where(mask)[0]

    def __getitem__(self, index):
        # t = time.time()
        size = min(self.batchSize, len(self.pts) - (index * self.batchSize))

        if(self.generateALl):
            size = len(self.pts)
            index = 0
            
        if self.cts.input_tile_count == 1:
            dims = [size, self.cts.npoints]
        elif self.cts.input_tile_count > 1:
            dims = [size, self.cts.input_tile_count, self.cts.npoints]
        
        ptsList = np.zeros(dims + [self.cts.pointComponents], np.float32)
        if not self.cts.noFeature:
            ftsList = np.zeros(dims + [self.cts.featureComponents], np.float32)        
        if(self.test):
            lblList = np.zeros(dims + [self.cts.classCount], np.uint8)
        else:
            lblList = []
        self.avg_pts_in_tile = 0        
        for i in (tqdm(range(size)) if (self.generateALl) else range(size)):
            # get the data
            center = self.pts[index*self.batchSize + i]
            mask = self.compute_mask(center, self.cts.blocksize, self.xyzrgblbl)
            ptsftslbl = self.xyzrgblbl[mask]
            self.avg_pts_in_tile += len(ptsftslbl)
            if(len(ptsftslbl) == 0):
                continue
            
            ptsftslbl[:,:3] = ptsftslbl[:,:3] - np.array([center[0], center[1], ptsftslbl[:,2].min()])
            # choose right number of points
            choice = RandomChoice(len(ptsftslbl), self.npoints)
            ptsftslbl = ptsftslbl[choice]

            # labels will contain indices in the original point cloud
            idx = mask[choice]
            self.idxList[index*self.batchSize + i] = np.expand_dims(idx, 0)
            
            # one-hot encode test labels
            if self.test:
                lblList[i] = np.eye(self.cts.classCount, dtype=np.uint8)[ptsftslbl[:,-1].astype(int)]
            # separate between features and points
            if not self.cts.noFeature:
                if(self.featureComponents == 1):
                    ftsList[i] = np.expand_dims(ptsftslbl[:,3], 1)
                else:
                    ftsList[i] = ptsftslbl[:,3:6]
            ptsList[i] = np.expand_dims(ptsftslbl[:, :3], 0)                           
            
        # print(f"Batch generated in {time() - t:.3f}")
        self.avg_pts_in_tile /= size

        add_lbl = [lblList] if self.test else []
        if self.nocolor:
            return [ptsList] + add_lbl
        else: #works for RGB
            return [[ftsList, ptsList]] + add_lbl
    
def GetValidationData(data_pipeline):
    ptsList = None
    ftsList = None
    lbsList = None

    for data in tqdm(data_pipeline):
        ftspts, lbl = data
        
        if(isinstance(ftspts, dict)):
            ftspts = [x.numpy() for x in ftspts.values()]
            
        pts, fts = ftspts if isinstance(ftspts, list) else (ftspts, None)
        
        if(isinstance(lbl, tf.Tensor)):
            lbl = lbl.numpy()
        
        ptsList = pts if (ptsList is None) else np.concatenate((ptsList, pts), 0)
        lbsList = lbl if (lbsList is None) else np.concatenate((lbsList, lbl), 0)
        
        if not(fts is None):            
            ftsList = fts if (ftsList is None) else np.concatenate((ftsList, fts), 0)

    return ((ptsList if (ftsList is None) else [ftsList, ptsList]), lbsList)
    
def SaveLabelsPnts(labels, outputFile):
    import pandas as pd    
    print("Saving pts lbs...")
    if(len(labels.shape) == 1):
        pd.DataFrame(labels, dtype=np.uint8).to_csv(outputFile, sep='\t', header=None, index=None)
    else:
        np.save(outputFile, labels)
    print("Pts lbs {} saved!".format(labels.shape))
    
def TestTestSequence(path, consts):    
    seq = TestSequence(path, consts)

    allPts = np.zeros((len(seq.xyzrgb), 3))

    for i in range(len(seq)):
        inpt = seq[i]

        ftsList = inpt[0]
        ptsList = inpt[1]

        for j in range(len(ptsList)):
            allPts[seq.idxList[i*consts.BatchSize() + j]] = ptsList[j]
    
    emptyPts = np.logical_not(allPts.sum(1) != 0)

    print("sparseCubes: ",seq.sparseCubes)
    print("mean sparseCubes pt count: ", seq.sparseCubesPtCount/seq.sparseCubes)
    print("Not picked points: {} => {:.2f}%".format(len(emptyPts), len(emptyPts)/len(allPts)))

    nonEmptyPts = np.logical_not(emptyPts)

    a = seq.xyzrgb[emptyPts]
    b = seq.xyzrgb[nonEmptyPts]

    dt = DataTool()
    dt.VisualizePointCloud([a, b], [[1,0,0], None])