from multiprocessing import Pool
import numpy as np
import tensorflow as tf

from multiprocessing import Pool
from functools import partial
from configs.Config import Config
import os

knn_dll_path = './knn_kdtree' + ('.dll' if Config.IsWindowsMachine() else '.so')
knn_farthest_dll_path = './sampling/farthest_sampling'+ ('.dll' if Config.IsWindowsMachine() else '.so')

knn_farthest_dll_exists = False
if(os.path.exists(knn_farthest_dll_path)):
    farthest_point_sampling_module = tf.load_op_library(knn_farthest_dll_path)
    farthest_point_sampling_op = farthest_point_sampling_module.farthest_point_sample    
    knn_farthest_dll_exists = True
    use_dll_knn = True  
if(os.path.exists(knn_dll_path)):
    knn_kdtree_module = tf.load_op_library(knn_dll_path)
    knn_kdtree_op = knn_kdtree_module.knn_kdtree
    knn_kdtree_sampler_op = knn_kdtree_module.knn_kdtree_sampler
    use_dll_knn = True
else:
    import knn.lib.python.nearest_neighbors as nearest_neighbors
    use_dll_knn = False    

# from scipy.spatial import cKDTree, KDTree
from sklearn.neighbors import KDTree
from time import time

class KDTreeLayer(tf.keras.layers.Layer):
    '''
    Input:\n
        pointCount\n
        points: (b, n, 3)\n
        newPoints: (b, m, 3)\n
    Output:\n
        idx: (b, m, pointCount)\n
    '''
    def __init__(self, pointCount, **kwargs):
        super(KDTreeLayer, self).__init__(**kwargs)
        self.pointCount = pointCount        
    
    # @tf.function(experimental_compile=True)
    @tf.function()
    def call(self, xyz, new_xyz):
        b = tf.shape(new_xyz)[0]
        m = new_xyz.get_shape()[1]

        # idx = tf.py_function(knn_kdtree, [self.pointCount, xyz, new_xyz, False], tf.int32)
        if use_dll_knn:
            idx = knn_kdtree_op(self.pointCount, xyz, new_xyz)
        else:
            idx = tf.py_function(cython_knn_kdtree, [self.pointCount, xyz, new_xyz], tf.int64, name = "KDtree")

        # idx = tf.cast(idx, tf.int32)
        idx = tf.reshape(idx, (b, m, self.pointCount))

        # pts = group_point(xyz, idx)        
        return idx
    
    def get_config(self):
        config = super(KDTreeLayer, self).get_config()
        config.update({'pointCount': self.pointCount})
        return config

class KDTreeSampleLayer(tf.keras.layers.Layer):
    '''
    Input:\n
        pointCount\n
        nqueries\n
        points: (b, n, 3)\n        
    Output:\n
        idx: (b, nqueries, pointCount, 2)\n
        pts: (b, nqueries, pointCount)\n
    '''
    def __init__(self, pointCount, nqueries, farthest_sampling = True, **kwargs):
        super(KDTreeSampleLayer, self).__init__(**kwargs)
        self.pointCount = pointCount       
        self.nqueries = nqueries
        self.farthest_sampling = farthest_sampling
    
    # @tf.function(experimental_compile=True)
    @tf.function()
    def call(self, xyz):
        b = tf.shape(xyz)[0]

        # idx, pts = tf.py_function(knn_kdtree, [self.pointCount, xyz, self.nqueries, True], [tf.int32, tf.float32])        
        if use_dll_knn:
            if(self.farthest_sampling):
                idx = farthest_point_sampling_op(xyz, self.nqueries)
                pts = tf.gather_nd(xyz, tf.expand_dims(idx, -1), batch_dims=1)
                idx = knn_kdtree_op(self.pointCount, xyz, pts)
            else:
                # pts =  xyz[:,:self.nqueries,:]
                # idx = knn_kdtree_op(self.pointCount, xyz, pts)
                idx, pts = knn_kdtree_sampler_op(self.pointCount, xyz, self.nqueries)
        else:
            idx, pts = tf.py_function(cython_knn_kdtree_sampler, [self.pointCount, xyz, self.nqueries], [tf.int64, tf.float32], name="knn_sample")

        # idx = tf.cast(idx, tf.int32)
        idx = tf.reshape(idx, (b, self.nqueries, self.pointCount))
        pts = tf.reshape(pts, (b, self.nqueries, 3))

        return idx, pts
    
    def get_config(self):
        config = super(KDTreeSampleLayer, self).get_config()
        config.update({'pointCount': self.pointCount, 'nqueries': self.nqueries, 'farthest_sampling':self.farthest_sampling})
        return config


def knn_kdtree(nsample, xyz, new_xyz, resample = False):
    # if isinstance(xyz, tf.Tensor):
    xyz = xyz.numpy()
        
    if resample:
        rplc = False
        if(xyz.shape[0]<new_xyz):
            rplc = True

        new_xyz = [xyz[i][np.random.choice(xyz.shape[1], new_xyz, replace = rplc)] for i in range(xyz.shape[0])]
        new_xyz = np.asarray(new_xyz)
    else:
        # if isinstance(new_xyz, tf.Tensor):
        new_xyz = new_xyz.numpy()

    batch_size = xyz.shape[0]
    n_points = new_xyz.shape[1]

    # indices = np.zeros((batch_size, n_points, nsample, 2), dtype=np.int32)
    indices = np.zeros((batch_size, n_points, nsample, 1), dtype=np.int32)

    for batch_idx in range(batch_size):
        X = xyz[batch_idx, ...]
        q_X = new_xyz[batch_idx, ...]
        kdt = KDTree(X, leaf_size=10) #CoinvPoint suggests 10
        _, batch_indices = kdt.query(q_X, k = nsample)

        ##### fill batch indices like in nearest neighbors layer
        # indicesForBatch = np.full((batch_indices.shape[0], batch_indices.shape[1]), batch_idx)
        # batch_indices = np.concatenate((np.expand_dims(indicesForBatch, axis=2), np.expand_dims(batch_indices, axis=2)), axis=2)
        batch_indices = np.expand_dims(batch_indices, axis=2)

        indices[batch_idx] = batch_indices
    
    if resample:
        return indices, new_xyz
    else:
        return indices

def cython_knn_kdtree(nsample, xyz, new_xyz):
    return nearest_neighbors.knn_batch(xyz, new_xyz, nsample, omp=True)

def cython_knn_kdtree_sampler(K, xyz, npts):
    return nearest_neighbors.knn_batch_distance_pick(xyz, npts, K, omp=True)

def multiproces_kdtree(nsample, xyz, new_xyz, resample = False):
    # if isinstance(xyz, tf.Tensor):
    # xyz = xyz.numpy()
            
    if resample: ######## move this to threads
        rplc = False
        if(xyz.shape[0]<new_xyz):
            rplc = True

        new_xyz = [xyz[i][np.random.choice(xyz.shape[1], new_xyz, replace = rplc)] for i in range(xyz.shape[0])]
        new_xyz = np.asarray(new_xyz)
    # else:
    #     # if isinstance(new_xyz, tf.Tensor):
    #     new_xyz = new_xyz.numpy()

    batch_size = xyz.shape[0]
    n_points = new_xyz.shape[1]

    indices = np.zeros((batch_size, n_points, nsample, 2), dtype=np.int32)

    data = [(xyz[batch_idx, ...], new_xyz[batch_idx, ...]) for batch_idx in range(batch_size)]
    pool = Pool(processes = 8)
    GenerateFunc=partial(ProcessBatch, nsample=nsample)
    result = pool.starmap(GenerateFunc, data)

    for batch_idx in range(batch_size):
        batch_indices = result[batch_idx]

        ##### fill batch indices like in nearest neighbors layer
        indicesForBatch = np.full((batch_indices.shape[0], batch_indices.shape[1]), batch_idx)
        batch_indices = np.concatenate((np.expand_dims(indicesForBatch, axis=2), np.expand_dims(batch_indices, axis=2)), axis=2)

        indices[batch_idx] = batch_indices
    
    if resample:
        return indices, new_xyz
    else:
        return indices

def ProcessBatch(X, q_X, nsample):
    kdt = KDTree(X, leaf_size=30) #CoinvPoint suggests 10
    _, batch_indices = kdt.query(q_X, k = nsample)

    return batch_indices

if __name__ == "__main__":
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv2D
    from tensorflow.python import debug as tf_debug
    from tensorflow.python.eager import profiler as profile
    from NearestNeighbors import NearestNeighborsLayer, SampleNearestNeighborsLayer

    profile.start_profiler_server(6009)
    tf.compat.v1.keras.backend.set_session(tf_debug.TensorBoardDebugWrapperSession(tf.compat.v1.Session(), "DESKTOP-TKAPBCU:6007"))
    log_dir="logs"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch = 3)

    batchSize = 1000
    pointCount = 5
    aCount = 2000
    radius = 1
    
    a = [[[9,9,9], [8,8,8], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], 
        [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1],
        [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2],
        [3, 3, 3], [3, 3, 3]]]
    a = np.array(a, dtype=np.float32)

    b = [[[9,9,9], [8,8,8], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]]
    b = np.array(b, dtype=np.float32)

    a = np.concatenate([np.zeros((1, aCount, 3), dtype=np.float32), a], axis=1)

    a = np.tile(a, (batchSize, 1, 1))
    b = np.tile(b, (batchSize, 1, 1))
    # a = np.tile(a, (3, 1, 1))
    # b = np.tile(b, (3, 1, 1))

    t=time()
    # RandomSample(a, 1000)
    # tf.print("Random sample done in {:.5f}".format((time() - t)/60))
    idx = knn_kdtree(5, a, b)
    print("knn_kdtree done:",time() - t)
    t=time()
    idx = multiproces_kdtree(5, a, b)
    print("multiproces_kdtree done:",time() - t)
    # print(idx)
    
    input()

    ptsA = Input(shape=(a.shape[1], 3), dtype=tf.float32)
    ptsB = Input(shape=(b.shape[1], 3), dtype=tf.int32)

    ps = tf.expand_dims(ptsA, axis = 1)    
    ps = Conv2D(3, [1, 1], [1, 1], 'valid', activation='relu')(ps)
    # ps = Conv2D(3, [1, 1], [1, 1], 'valid', activation='relu')(ps)
    # ps = Conv2D(3, [1, 1], [1, 1], 'valid', activation='relu')(ps)
    # ps = Conv2D(3, [1, 1], [1, 1], 'valid', activation='relu')(ps)
    # ps = Conv2D(3, [1, 1], [1, 1], 'valid', activation='relu')(ps)
    # ps = Conv2D(3, [1, 1], [1, 1], 'valid', activation='relu')(ps)
    # ps = Conv2D(3, [1, 1], [1, 1], 'valid', activation='relu')(ps)
    # ps = Conv2D(3, [1, 1], [1, 1], 'valid', activation='relu')(ps)
    # ps = Conv2D(3, [1, 1], [1, 1], 'valid', activation='relu')(ps)
    ps = tf.squeeze(ps, axis=1)

    # out = KDTreeLayer(5)(ps, ptsB)
    # out, outpts = KDTreeSampleLayer(5, 1000)(ps)
    out, outpts = SampleNearestNeighborsLayer(5, 1000)(ps)
    out = tf.cast(out, tf.float32)

    # out = tf.expand_dims(out, axis = 1)
    out = Conv2D(5, [1, 1], [1, 1], 'valid', activation='relu')(out)
    # out = Conv2D(5, [1, 1], [1, 1], 'valid', activation='relu')(out)
    # out = Conv2D(5, [1, 1], [1, 1], 'valid', activation='relu')(out)
    # out = Conv2D(5, [1, 1], [1, 1], 'valid', activation='relu')(out)
    # out = Conv2D(5, [1, 1], [1, 1], 'valid', activation='relu')(out)
    # out = Conv2D(5, [1, 1], [1, 1], 'valid', activation='relu')(out)
    # out = Conv2D(5, [1, 1], [1, 1], 'valid', activation='relu')(out)
    # out = Conv2D(5, [1, 1], [1, 1], 'valid', activation='relu')(out)
    # out = Conv2D(5, [1, 1], [1, 1], 'valid', activation='relu')(out)
    # out = Conv2D(5, [1, 1], [1, 1], 'valid', activation='relu')(out)    
    # out = tf.squeeze(out, axis=1)

    model = Model([ptsA, ptsB], [out], name ="model")
    model.compile(tf.keras.optimizers.Adam(), loss='mse', metrics=['accuracy'])

    print(model.summary())

    y = np.ones((a.shape[0], 1000, 5, 5), dtype=np.float32)
    # y = np.ones((batchSize, 1000, 5), dtype=np.float32)
    model.fit([a, b], [y], batch_size = 100, epochs = 3, callbacks=[tensorboard_callback])