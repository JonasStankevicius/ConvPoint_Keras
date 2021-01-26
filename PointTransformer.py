from imports import *
from dataTool import DataTool
from ConvPoint import TrainSequence, Semantic3D, CompileModel

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Dense, Dropout, InputLayer


def MLP(points, features, dim, pos_mlp_hidden_dim, attn_mlp_hidden_mult):
    return None

def TransitionDown(pos, features):
    
    return None

def TransitionUp(pos, features):
    return None

def PointTransformer(pos, features, dim, pos_mlp_hidden_dim, attn_mlp_hidden_mult):
    n = features.shape[1]

    # get queries, keys, values
    qkv = Dense(dim*3, use_bias = False)(features)
    q, k, v = tf.split(qkv, 3, axis=-1)

    # calculate relative positional embeddings
    rel_pos = pos[:, :, None] - pos[:, None, :]
    rel_pos_emb = Dense(pos_mlp_hidden_dim, activation="relu")(rel_pos)
    rel_pos_emb = Dense(dim) (rel_pos_emb)

    # use subtraction of queries to keys. i suppose this is a better inductive bias for point clouds than dot product
    qk_rel = q[:, :, None] - k[:, None, :]

    # use attention mlp, making sure to add relative positional embedding first    
    sim = Dense(dim * attn_mlp_hidden_mult, activation="relu")(qk_rel + rel_pos_emb)
    sim = Dense(dim)(sim)

    # expand transformed features and add relative positional embeddings
    v = tf.tile(tf.expand_dims(v, axis=1), [1,n,1,1])
    v = v + rel_pos_emb

    # attention
    attn = tf.nn.softmax(sim, -2)

    # aggregate
    agg = tf.einsum('b i j d, b i j d -> b i d', attn, v)
    return agg

def CreateModel(pointsCount, classCount):

    in_pts = Input(shape=(pointsCount, 3), dtype=tf.float32) # points 
    in_fts = tf.ones((pointsCount, 1), dtype=tf.float32) # dummy features until we dont use point colors

    # Block 1
    ft0 = Dense(16)(in_fts)
    ft1 = PointTransformer(in_pts, ft0)

    # Block 2



    model = Model(inputList, out_labels, name ="model")    
    model = CompileModel(model, classCount)
    return model



if __name__ == "__main__":
    from NearestNeighbors import NearestNeighborsLayer, SampleNearestNeighborsLayer
    from KDTree import KDTreeLayer, KDTreeSampleLayer

    # x = tf.random.uniform((5, 16, 128))
    # pos = tf.random.uniform((5, 16, 3))
    # out = PointTransformer(pos, x, 128, 64, 4)


    # for the first test semantic3d dataset will be used    
    classes = 8
    points = 8192

    # create model
    model = CreateModel()

    # get dataset samples generator
    consts = Semantic3D()
    seq = TrainSequence(consts.TrainFiles(), 200, consts)

    # fit model
    model.fit(seq, epochs = 10)