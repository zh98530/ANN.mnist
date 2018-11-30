#coding:utf-8

import tensorflow as tf

#照片为28*28像素，数据输入量为784,输出显示为0-9这十个数字，隐藏层节点个数为500
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

def get_weight(shape,regularizer):
	w = tf.Variable(tf.random_normal(shape),dtype=tf.float32)
	if regularizer != None: 
		tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
	return w

def get_bias(shape):
	b = tf.Variable(tf.constant(0.01,shape=shape))
	return b

def forward(x,regularizer):
	w1 = get_weight([INPUT_NODE,LAYER1_NODE],regularizer)
	b1 = get_bias([LAYER1_NODE])
	y1 = tf.nn.relu(tf.matmul(x,w1) + b1)
	
	w2 = get_weight([LAYER1_NODE,OUTPUT_NODE],regularizer)
	b2 = get_bias([OUTPUT_NODE])
	y = tf.matmul(y1,w2) + b2
	return y
