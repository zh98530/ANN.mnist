#coding:utf-8
#coding:utf-8
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backward
TEST_INTERVAL_SECS = 5

def test(mnist):
	#复现训练完成的神经网络计算图
	with tf.Graph().as_default() as g:
		x = tf.placeholder(tf.float32,[None,mnist_forward.INPUT_NODE])
		y_ = tf.placeholder(tf.float32,[None,mnist_forward.OUTPUT_NODE])
		y = mnist_forward.forward(x,None)

		#实例化saver对象，实现参数滑动平均值的加载
		ema = tf.train.ExponentialMovingAverage(mnist_forward)
		ema_restore = ema.variables_to_restore()
		saver = tf.train.Saver(ema_restore)
		
		#将BATCH_SIZE组数据，每一组数据取0-9中概率最大值，返回其标签，即0-9，与标准答案比对，相等返回true，不等返回false;将得到的布尔型数值转化为实数型，并将其取平均作为本组数据准确率
		correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
		
		while True:
			#加载训练完成的模型
			with tf.Session() as sess:
				ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
				if ckpt and ckpt.model_checkpoint_path:
					#加载保存的会话
					saver.restore(sess,ckpt.model_checkpoint_path)
					
					#提取轮数
					global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
					
					#将训练集输入得数据准确率
					accuracy_score = sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
					print("After %s training step(s), test accuracy = %g." %(global_step,accuracy_score))
				else:
					print('No checkpoints file found')
					return
			#定义每一轮进行5s延迟
			time.sleep(TEST_INTERVAL_SECS)
	
def main():
	mnist = input_data.read_data_sets("./data/",one_hot=True)
	test(mnist)
	
if __name__ == '__main__':
	main()
