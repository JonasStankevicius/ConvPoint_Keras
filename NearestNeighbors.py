import numpy as np
import timeit
from time import time
import tensorflow as tf

class NearestNeighborsLayer(tf.keras.layers.Layer):
    '''
        Input:\n 
            xyz1 (b, n, 3), 
            xyz2 (b, m, 3), 
            neighbors (1) \n
        Output:\n 
            idx (b, m, neighbors, 2) np.int32
    '''    
    def __init__(self, neighbors, **kwargs):        
        super(NearestNeighborsLayer, self).__init__(**kwargs)
        self.neighbors = neighbors
    
    def call(self, xyz1, xyz2):
        return self.NearestNeighbors(xyz1, xyz2)
    
    def get_config(self):
        base_config = super(NearestNeighborsLayer, self).get_config()
        config = {'neighbors': self.neighbors}
        return dict(list(base_config.items()) + list(config.items()))

    @tf.function
    def NearestNeighbors(self, xyz1, xyz2):
        xyz1 = tf.stop_gradient(xyz1)
        xyz2 = tf.stop_gradient(xyz2)

        b = tf.shape(xyz1)[0]
        m = tf.shape(xyz2)[1]

        neighbors = self.neighbors

        idx = tf.zeros((b, m, neighbors, 1), dtype = tf.int32)
        # idx = tf.zeros((b, m, neighbors, 2), dtype = tf.int32)

        for i in tf.range(b):
            xyzArr1 = tf.squeeze(tf.gather_nd(xyz1, [[i]]), axis = 0)
            xyzArr2 = tf.squeeze(tf.gather_nd(xyz2, [[i]]), axis = 0)

            ptsArr = tf.math.squared_difference(xyzArr1, tf.expand_dims(xyzArr2, axis = 1))
            ptsArr = tf.reduce_sum(ptsArr, axis=2)

            index = tf.nn.top_k(tf.negative(ptsArr), k = neighbors, sorted = True)[1]
            index = tf.expand_dims(index, 2)
            # batchIndexes = tf.fill([index.shape[0], index.shape[1]], i)
            # index = tf.concat([tf.expand_dims(batchIndexes,2), tf.expand_dims(index,2)], axis = 2)

            idx = tf.tensor_scatter_nd_update(idx, [[i]], tf.expand_dims(index,axis=0))

        return idx

# a = np.array([[[3.5,3.5,3.5], [1,1,1], [2.5,2.5,2.5]], [[4,4,4], [3.5,3.5,3.5], [4,4,4]], [[6,6.5,6], [5,5,5], [5,5,5]]], dtype=np.float32)
# b = np.array([[[1,1,1], [2.5,2.5,2.5]], [[3.5,3.5,3.5], [4,4,4]], [[5,5,5], [6,6.5,6]]], dtype=np.float32)
# result = NearestNeighborsLayer(2).NearestNeighbors(a, b)

class SampleNearestNeighborsLayer(tf.keras.layers.Layer):
    '''
        Input:\n 
            xyz1 (b, n, 3), \n
            neighbors (1) \n
            nqueries (1)\n
        Output:\n 
            idx: (b, m, neighbors, 2) np.int32 \n
            pts: (b, nqueries, pointCount)\n
    '''    
    def __init__(self, neighbors, nqueries, **kwargs):        
        super(SampleNearestNeighborsLayer, self).__init__(**kwargs)
        self.neighbors = neighbors
        self.nqueries = nqueries
    
    def call(self, xyz1, legacy = False):
        if(legacy):
            return self.SampleNearestNeighbors(xyz1)    

        return self.indices_conv_reduction(xyz1)
    
    def get_config(self):
        config = super(SampleNearestNeighborsLayer, self).get_config()
        config.update({'neighbors': self.neighbors, 'nqueries': self.nqueries})
        return config

    @tf.function
    def SampleNearestNeighbors(self, xyz1):
        xyz1 = tf.stop_gradient(xyz1)

        b = tf.shape(xyz1)[0]
        n = xyz1.shape[1]

        neighbors = self.neighbors
        nqueries = self.nqueries

        idx = tf.zeros((b, nqueries, neighbors, 1), dtype = tf.int32)
        # idx = tf.zeros((b, nqueries, neighbors, 2), dtype = tf.int32)
        pts = tf.zeros((b, nqueries, 3), dtype = tf.float32)

        for i in tf.range(b):
            xyzArr1 = tf.squeeze(tf.gather_nd(xyz1, [[i]]), axis = 0)
            
            # randIndices = tf.random.uniform_candidate_sampler(tf.expand_dims(tf.range(n, dtype=tf.int64),0), n, nqueries, True, nqueries)[0]
            randIndices = tf.random.shuffle(tf.range(n, dtype=tf.int64))[:nqueries]
            randIndices = tf.concat([tf.cast(tf.fill((nqueries, 1), i), tf.int64), tf.expand_dims(randIndices, axis=1)], axis = 1)
            xyzArr2 = tf.gather_nd(xyz1, randIndices)

            ptsArr = tf.math.squared_difference(xyzArr1, tf.expand_dims(xyzArr2, axis = 1))
            ptsArr = tf.reduce_sum(ptsArr, axis=2)

            index = tf.nn.top_k(tf.negative(ptsArr), k = neighbors, sorted = True)[1]
            index = tf.expand_dims(index, 2)
            # batchIndexes = tf.fill([index.shape[0], index.shape[1]], i)
            # index = tf.concat([tf.expand_dims(batchIndexes,2), tf.expand_dims(index,2)], axis = 2)            

            idx = tf.tensor_scatter_nd_update(idx, [[i]], tf.expand_dims(index,axis=0))
            pts = tf.tensor_scatter_nd_update(pts, [[i]], tf.expand_dims(xyzArr2,axis=0))

        return idx, pts

    @tf.function
    def indices_conv_reduction(self, pts):
        pts = tf.stop_gradient(pts)

        b = tf.shape(pts)[0]
        n = tf.shape(pts)[1]

        K = self.neighbors
        npts = self.nqueries

        indices = tf.zeros((b, npts, K, 1), dtype = tf.int32)
        points = tf.zeros((b, npts, 3), dtype = tf.float32)

        for ib in tf.range(b):

            used = tf.zeros(n, tf.int32)
            current_id = 0

            xyz = tf.squeeze(tf.gather_nd(pts, [[ib]]), 0)

            for i in tf.range(npts):
                possible_ids = tf.where(tf.equal(used, current_id))

                if(len(possible_ids) == 0):
                    current_id = tf.keras.backend.min(used)
                    possible_ids = tf.where(tf.equal(used, current_id))
                
                possible_ids = tf.squeeze(possible_ids, 1)

                rand_int = int(tf.random.uniform((1,), 0, len(possible_ids), tf.int32))
                index = int(tf.gather_nd(possible_ids, [rand_int]))

                pt = tf.gather_nd(xyz, [index])

                ptsArr = tf.math.squared_difference(xyz, pt)
                ptsArr = tf.reduce_sum(ptsArr, axis=1)
                ids = tf.nn.top_k(tf.negative(ptsArr), k = K, sorted = True)[1]
                    
                used = tf.tensor_scatter_nd_update(used, tf.expand_dims(ids,1), tf.gather_nd(used, tf.expand_dims(ids,1))+1)
                used = tf.tensor_scatter_nd_update(used, [[index]], [tf.gather_nd(used, [index])+100])

                indices = tf.tensor_scatter_nd_update(indices, [[ib, i]], [tf.expand_dims(ids,1)])
                points = tf.tensor_scatter_nd_update(points, [[ib, i]], pt)

        return indices, points

if __name__ == "__main__":    
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv2D, Conv1D, BatchNormalization, Dropout, Softmax, Dense
    from tensorflow.python import debug as tf_debug
    from tensorflow.python.eager import profiler as profile
    from KDTree import KDTreeLayer

    # profile.start_profiler_server(6009)
    # tf.compat.v1.keras.backend.set_session(tf_debug.TensorBoardDebugWrapperSession(tf.compat.v1.Session(), "DESKTOP-TKAPBCU:6007"))
    # log_dir="logs"
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch = 3)

    ### MODEL PART
    batchSize = 1000
    neighbors = 5
    aCount = 2000
    bCount = 1000
    radius = 1

    # temp = np.zeros((1, aCount, 3), dtype=np.float32)
    a = [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], 
        [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1],
        [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2],
        [3, 3, 3], [3, 3, 3], [5, 4, 6], [7, 7, 9], [1, 2, 1]]]
    a = np.array(a, dtype=np.float32)
    # a = np.tile(a, (2, 1, 1))
    a = np.tile(a, (1, 100, 1))
    a = np.tile(a, (batchSize, 1, 1))

    # a = np.concatenate([temp, a], axis=1)

    x = np.ones((batchSize, a.shape[1], neighbors), dtype=np.int32)
    
    # haltt = time()
    idx = NearestNeighborsLayer(neighbors)(a, a)
    # print("NearestNeighborsLayer done in {}:{} min.".format(int((time() - haltt)/60), int((time() - haltt)%60)))
    # haltt = time()
    # idx2 = KDTreeLayer(neighbors)(a, a)
    # print("KDTreeLayer done in {}:{} min.".format(int((time() - haltt)/60), int((time() - haltt)%60)))

    # idx = NearestNeighbors(a, a, neighbors)
    # idx, pts = SampleNearestNeighbors(a, 5, 3)

    inA = Input(shape=(a.shape[1], 3), dtype=tf.float32)
    inB = Input(shape=(a.shape[1], 3), dtype=tf.float32)

    out = tf.expand_dims(inA, axis = 1)    
    out = Conv2D(3, [1, 1], [1, 1], 'valid', activation='relu')(out)
    out = tf.squeeze(out, axis=1)

    idx = NearestNeighborsLayer(neighbors)(out, inB)
    # idx = KDTreeLayer(neighbors)(inA, inB)
    idx = tf.cast(idx, tf.float32)
    idx = Dense(1)(idx)
    idx = tf.squeeze(idx, 3)
    

    out = tf.expand_dims(idx, axis = 1)    
    out = Conv2D(5, [1, 1], [1, 1], 'valid', activation='relu')(out)
    idx = tf.squeeze(out, axis=1)

    model = Model([inA, inB], idx, name = "model")
    model.compile(tf.keras.optimizers.Adam(), loss='mse', metrics=['accuracy'])

    print(model.summary())

    # model.fit([a, a], [x], batch_size = 10, epochs = 1, callbacks=[tensorboard_callback])
    model.fit([a, a], [x], batch_size = 100, epochs = 2)