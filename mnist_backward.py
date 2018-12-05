#coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import os

STEPS = 50000
BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "mnist_model"

def backward(mnist):
	x = tf.placeholder(tf.float32,[None,mnist_forward.INPUT_NODE])
	y_ = tf.placeholder(tf.float32,[None,mnist_forward.OUTPUT_NODE])
	y = mnist_forward.forward(x,REGULARIZER)
	global_step = tf.Variable(0,trainable=False)
	
	#前向传播输出经过softmax函数，以获得输出分类的概率分布，与标准答案求出交叉熵，加上正则化权重得损失函数
	ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
	cem = tf.reduce_mean(ce)
	loss = cem + tf.add_n(tf.get_collection("losses"))
	
	#学习率定为指数衰减型
	learning_rate = tf.train.exponential_decay(
		LEARNING_RATE_BASE,
		global_step,
		mnist.train.num_examples/BATCH_SIZE,
		LEARNING_RATE_DECAY,
		staircase=True)
	
	#采取梯度下降优化	
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
	
	#滑动平均更新新的神经网络参数，保留上一次的参数影响，给予其合适的权重影响新参数
	ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
	ema_op = ema.apply(tf.trainable_variables())
	with tf.control_dependencies([train_step,ema_op]):
		train_op = tf.no_op(name = 'train')
	
	#实例化saver对象	
	saver = tf.train.Saver()
	
	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		
		for i in range(STEPS):
			xs,ys = mnist.train.next_batch(BATCH_SIZE)#训练参数
			_,loss_value,step = sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})#参数喂入得到相应的评估参数
			if i % 1000 == 0:
				print("After %d training step(s),loss on training batch is %g." %(step,loss_value)) 
				saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)#将当前sess保存至指定路径，并命名mnist_model_(global_step)

def main():
	mnist = input_data.read_data_sets("./data/",one_hot=True)
	backward(mnist)

if __name__ == '__main__':
	main()
