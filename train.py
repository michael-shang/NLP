import tensorflow as tf
from data_pre_process import *
#import Reader
#import collections
#import sys
#import re
#import argparse
#from langconv import *
#from tensorflow.contrib import rnn
#import numpy as np
import matplotlib.pyplot as plt


embedding_size = 300
vocabulary_size = 20000
batch_size = 50
max_sequence_len = 20
num_hidden = 100
n_output = 100
n_classes = 3#####
train_dropout_keep_prob = 0.8
test_dropout_keep_prob = 1
learning_rate = 0.00025
epochs = 250
log_file = '/tmp/short_text_stance'

train_file = './Data/Chinese/short_conversation_corpus/stanceResponse.train'
test_file = './Data/Chinese/short_conversation_corpus/stanceResponse.test'

def biLSTM(num_hidden, dropout_keep_prob, input_embed_lookup_data, name = 'biLSTM'):
	#cell_unit = tf.nn.rnn_cell.BasicLSTMCell
	with tf.name_scope(name):
		#forward direction layer
		with tf.name_scope("FW"):
			lstm_forward_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias = 1.0)
			lstm_forward_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_forward_cell, 
													output_keep_prob = dropout_keep_prob)
		#backward direction layer
		with tf.name_scope("BW"):
			lstm_backward_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias = 1.0)
			lstm_backward_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_backward_cell, 
													output_keep_prob = dropout_keep_prob)
						
		#input_embed_lookup_data = tf.unstack(input_embed_lookup_data, max_sequence_len, 1)
		try:
			(out_puts, states) = tf.nn.bidirectional_dynamic_rnn(lstm_forward_cell,
												lstm_backward_cell,
												input_embed_lookup_data,
												dtype = tf.float32)
		except Exception:
			out_puts = tf.nn.bidirectional_dynamic_rnn(lstm_forward_cell,
												lstm_backward_cell,
												input_embed_lookup_data,
												dtype = tf.float32)
		return out_puts
	#temporal_mean = tf.add_n(outputs) / input_length
	
reader_train = Reader.reader(train_file, language = 'chn')
#reader_test = Reader.reader(test_file, language='chn')
sents_train = reader_train.getWholeText()
#sents_test = reader_test.getWholeText()
posts, responses, labels = reader_train.getData()
#print("************labels**************", labels[:20])
#post_response_label = mergePostResponseLabel(posts, responses, labels)
word_dictionary, word_dictionary_rev = build_dictionary(posts + responses, labels, vocabulary_size)
posts_data = text2num(posts, word_dictionary)
responses_data = text2num(responses, word_dictionary)
y = generate_num_label(labels)

#split data into test & train set
ix_cutoff = int(len(labels) * 0.8)
train_posts_data, test_posts_data = posts_data[:ix_cutoff], posts_data[ix_cutoff:]
train_responses_data, test_responses_data = responses_data[:ix_cutoff], responses_data[ix_cutoff:]
train_y, test_y = y[:ix_cutoff], y[ix_cutoff:]
#print('train_posts_data******\n', train_posts_data[:50])

#add padding to sentence into max_time_steps 
train_posts_data_padding = append_padding(train_posts_data, max_sequence_len)
train_responses_data_padding = append_padding(train_responses_data, max_sequence_len)
test_posts_data_padding = append_padding(test_posts_data, max_sequence_len)
test_responses_data_padding = append_padding(test_responses_data, max_sequence_len)

	
"""tensorflow vocab_processor, text to num..."""
	#vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequence_len)
	#print("*********************\n", list(vocab_processor.fit_transform(train_posts_data[0])))
	#print("*********************\n", vocab_processor.fit_transform(train_posts_data[:20]))
	
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

weights = {
		'post': tf.Variable(tf.random_normal([2 * num_hidden, n_output])),
		'response': tf.Variable(tf.random_normal([2 * num_hidden, n_output])),
		'fully_connectted': tf.Variable(tf.random_normal([4 * num_hidden, n_output]), name = 'fully_connectted_W'),
		'outs': tf.Variable(tf.random_normal([n_output, n_classes]), name = 'softmax_W')
}
biases = {
		'post': tf.Variable(tf.random_normal([n_output])),
		'response': tf.Variable(tf.random_normal([n_output])),
		'fully_connectted': tf.Variable(tf.random_normal([n_output]), name = 'fully_connectted_B'),
		'outs': tf.Variable(tf.random_normal([n_classes]), name = 'fully_connectted_B')
}
#placeholder
posts_data_input = tf.placeholder(tf.int32, [None, max_sequence_len], name = 'posts_input')
#posts_data_input = tf.placeholder(tf.int32, [batch_size, max_sequence_len])
responses_data_input = tf.placeholder(tf.int32, [None, max_sequence_len], name = 'responses_input')
y_target = tf.placeholder(tf.int32, [None], name = 'label')
dropout_keep_prob = tf.placeholder(tf.float32, name = 'drop_out')

posts_lookup_data = tf.nn.embedding_lookup(embeddings, posts_data_input)
responses_lookup_data = tf.nn.embedding_lookup(embeddings, responses_data_input)
with tf.variable_scope('biLSTM-posts'):
	posts_outputs = biLSTM(num_hidden, dropout_keep_prob, posts_lookup_data, name = 'none')
with tf.variable_scope('biLSTM-responses'):
	responses_outputs = biLSTM(num_hidden, dropout_keep_prob, responses_lookup_data, name = 'none')
#posts_responses_outputs = tf.concat(2, [posts_outputs[-1], responses_outputs[-1]])
posts_outs = tf.concat(posts_outputs, 2)
		#print(posts_outs.get_shape())
responses_outs = tf.concat(responses_outputs, 2)
		#print(responses_outs.get_shape())
with tf.variable_scope('model_out'):	
	outs = tf.concat([posts_outs, responses_outs], 2)#shape(batch, max_time_steps, 4 * num_hidden)
outs = tf.transpose(outs, [1, 0, 2])
last = tf.gather(outs, int(outs.get_shape()[0]) - 1)#outs[-1] is also ok, shape（batch, 4 * num_hidden）
#double fully connectted into n_classes, for drop too much character
logits_outs = tf.matmul(last, weights['fully_connectted']) + biases['fully_connectted']
logits_out = tf.matmul(logits_outs, weights['outs']) + biases['outs']
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits_out, labels = y_target)#label is dense vector
#losses = tf.nn.softmax_cross_entropy_with_logits(logits = logits_out, labels = _y_target_)#label must be one-hot (sparse)vector
with tf.variable_scope('loss'):
	loss = tf.reduce_mean(losses)
with tf.variable_scope('accuracy'):
	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_out, 1), tf.cast(y_target, tf.int64)), tf.float32))
optimizer = tf.train.RMSPropOptimizer(learning_rate)
##########train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)#####最速梯度下降
train_step = optimizer.minimize(loss)
tf.summary.scalar("loss",loss)
tf.summary.scalar("accuracy",accuracy)
tf.summary.histogram('fc_weights', weights['fully_connectted'])
tf.summary.histogram('fc_biases', biases['fully_connectted'])
tf.summary.histogram('nc_weights', weights['outs'])
tf.summary.histogram('nc_biases', biases['outs'])
merged_summary = tf.summary.merge_all()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	summary_writer = tf.summary.FileWriter(log_file, graph=tf.get_default_graph())
	train_average_losses = []
	train_average_accuracy = []
	test_loss = []
	test_accuracy = []
	for epoch in range(epochs):
		average_loss = 0
		average_accuracy_tmp = 0
		##Shuffle training data
		shuffled_id = np.random.permutation(np.arange(len(train_posts_data_padding)))
		train_posts_data_padding = np.array(train_posts_data_padding)[shuffled_id]
		train_responses_data_padding = np.array(train_responses_data_padding)[shuffled_id]
		train_y = np.array(train_y)[shuffled_id]
		
		num_batches = int(len(train_posts_data_padding) / batch_size) + 1   
		for i in range(num_batches):
			min_ix = i * batch_size
			max_ix = np.min([len(train_posts_data_padding), ((i + 1) * batch_size)])
			x_posts_batches = train_posts_data_padding[min_ix:max_ix]
			x_responses_batches = train_responses_data_padding[min_ix:max_ix]
			y_label_batches = train_y[min_ix:max_ix]	

			train_dict = {posts_data_input: x_posts_batches, 
							responses_data_input: x_responses_batches,
							y_target: y_label_batches,
							dropout_keep_prob: train_dropout_keep_prob}
			sess.run(train_step, feed_dict = train_dict)
			summary = sess.run(merged_summary, feed_dict = train_dict)
			summary_writer.add_summary(summary,  epoch * num_batches + i)
		
			batch_loss, batch_accuracy = sess.run([loss, accuracy], feed_dict = train_dict)
			average_loss += batch_loss / num_batches
			average_accuracy_tmp += batch_accuracy / num_batches
		#train loss and accuracy
		train_average_losses.append(average_loss)
		train_average_accuracy.append(average_accuracy_tmp)
		if (epoch + 1) % 10 == 0:
			print("****epoch: {}, train loss: {}****".format(epoch + 1, average_loss))
			print("*************, train accuracy: {}****".format(average_accuracy_tmp))
		
		#run eval step
		test_dict = {posts_data_input: test_posts_data_padding, 
						responses_data_input: test_responses_data_padding,
						y_target: test_y,
						dropout_keep_prob: test_dropout_keep_prob}
		test_loss_temp, test_accuracy_temp = sess.run([loss, accuracy], feed_dict = test_dict)
		test_loss.append(test_loss_temp)
		test_accuracy.append(test_accuracy_temp)
		if (epoch + 1) % 10 == 0:
			print("************** test loss: {}****".format(test_loss_temp))
			print("************** test accuracy: {}****".format(test_accuracy_temp))
# Plot train loss over time
plt.plot(train_average_losses, 'k--')
plt.title('train loss')
plt.xlabel('generation')
plt.ylabel('loss')
plt.show()

# Plot train accuracy over time
plt.plot(train_average_accuracy, 'r-')
plt.title('train accuracy')
plt.xlabel('generation')
plt.ylabel('accuracy')
plt.show()

# Plot test loss over time
plt.plot(test_loss, 'k--')
plt.title('test loss')
plt.xlabel('generation')
plt.ylabel('loss')
plt.show()

# Plot test accuracy over time
plt.plot(test_accuracy, 'r-')
plt.title('test accuracy')
plt.xlabel('generation')
plt.ylabel('accuracy')
plt.show()






#train(path)