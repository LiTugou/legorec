from layers import *
import tensorflow as tf

bs=64
field_num=23
emb_size=16
inputs=tf.ones((bs,field_num,emb_size))
MLP(out_dim=3,hidden_units=[128,64])(inputs)
SELayer([128])(inputs)

## Dinattention
query=tf.ones((bs,emb_size))
key=tf.ones((bs,10,emb_size))
seqlen=tf.ones((bs))*5
DinAttention()(key,query,seqlen)

## ppnet
inputs=tf.ones((bs,1288))
gate_emb=tf.ones((bs,128))
PPNetLayer(out_dim=64,hidden_units=[128],gate_hidden_units=[128,128])(inputs,gate_emb)

## fibi
inputs=tf.ones((64,23,16))
BilinearInteraction()(inputs)

## cin
inputs=tf.ones((64,23,16))
CinLayer([128,128])(inputs)

## Fm
inputs=tf.ones((64,23,16))
FMCrossLayer()(inputs)

## MMoE
inputs=tf.ones((bs,1288))
MMoELayer(4,[256],2,[128])(inputs)

## PLE
inputs=tf.ones((bs,1288))
PLELayer(1,4,[256],2,[128],1)(inputs)
