import numpy as np
import tensorflow as tf
from parameter import *

class VideoCaptionGenerator():
    def __init__(self, video_size, video_step, caption_size, caption_step, hidden_size, batch_size, output_keep_prob, init_bias_vector , pretrained_embd):
        #parameters
        self.video_size = video_size
        self.video_step = video_step
        self.caption_size = caption_size
        self.caption_step = caption_step
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.output_keep_prob = output_keep_prob
        self.init_bias_vector = init_bias_vector
        #encode_layers
        with tf.device("/cpu:0"):
            if Embd_flag is False:
                self.caption_encode_W = tf.Variable(tf.random_uniform([caption_size, hidden_size], -0.1, 0.1), name="caption_encode_W")
            else :
                self.caption_encode_W = tf.Variable(pretrained_embd, name="caption_encode_W")
        self.video_encode_W = tf.Variable(tf.random_uniform([video_size, hidden_size], -0.1, 0.1), name="video_encode_W")
        self.video_encode_b = tf.Variable(tf.zeros([hidden_size]), name="video_encode_b")        
        #lstm_layers
        self.lstm_1 = tf.contrib.rnn.LSTMCell(self.hidden_size, use_peepholes=True, state_is_tuple=True)
        self.lstm_1_dropout = tf.contrib.rnn.DropoutWrapper(self.lstm_1, output_keep_prob=self.output_keep_prob)
        #decode_layers
        self.caption_decode_W = tf.Variable(tf.random_uniform([hidden_size, caption_size], -0.1, 0.1), name="caption_decode_W")
        self.caption_decode_b = tf.Variable(self.init_bias_vector.astype(np.float32), name="caption_decode_b")
        #attension_layers
        self.video_attention_W = tf.Variable(tf.random_uniform([hidden_size, hidden_size], -0.1, 0.1), name="video_attention_W")
        self.video_attention_b = tf.Variable(tf.zeros([hidden_size]), name="video_attention_b")
        self.e_attention_W = tf.Variable(tf.random_uniform([hidden_size, hidden_size], -0.1, 0.1), name="e_attention_W")
        self.e_attention_w = tf.Variable(tf.random_uniform([hidden_size, 1], -0.1, 0.1), name="e_attention_w")        
    def buildModel(self):
        #return_tensors
        tf_video_array = tf.placeholder(tf.float32, [self.batch_size, self.video_step, self.video_size])
        tf_video_array_mask = tf.placeholder(tf.float32, [self.batch_size, self.video_step])
        tf_max_prob_index = tf.ones([self.batch_size], dtype=tf.int32)
        tf_caption_array = tf.placeholder(tf.int32, [self.batch_size, self.caption_step])
        tf_caption_array_mask = tf.placeholder(tf.float32, [self.batch_size, self.caption_step])
        tf_sampling_choice = tf.placeholder(tf.int32, [])
        tf_loss = 0.0        
        #encode_tensors
        video_array_flat = tf.reshape(tf_video_array, [-1, self.video_size])
        video_embed = tf.matmul(video_array_flat, self.video_encode_W) + self.video_encode_b
        video_embed = tf.reshape(video_embed, [self.batch_size, self.video_step, self.hidden_size])
        video_embed = tf.transpose(video_embed, [1, 0, 2])        
        video_embed_part = tf.reshape(video_embed, [-1, self.hidden_size])
        video_embed_part = tf.matmul(video_embed_part, self.video_attention_W) + self.video_attention_b 
        video_embed_part = tf.reshape(video_embed_part, [self.video_step, self.batch_size, self.hidden_size])        
        #state_tensors
        c_init = tf.zeros([self.batch_size, self.hidden_size])
        m_init = tf.zeros([self.batch_size, self.hidden_size])    
        #attension_tensors
        h_init = tf.zeros([self.batch_size, self.hidden_size])
        with tf.variable_scope("RNN"):
            for step in range(self.caption_step-1):
                if step == 0:
                    state_1 = (c_init, m_init)
                    output_1 = h_init        			
                else:
                    tf.get_variable_scope().reuse_variables()
                #attention_embed        		
                e = tf.tanh(tf.matmul(output_1, self.e_attention_W) + video_embed_part)
                e = tf.reshape(e, [-1, self.hidden_size])
                e = tf.matmul(e, self.e_attention_w)
                e = tf.reshape(e, [self.video_step, self.batch_size])
                e_hat_exp = tf.multiply(tf.transpose(tf_video_array_mask), tf.exp(e))
                denominator = tf.reduce_sum(e_hat_exp, 0)
                denominator_regularized = denominator + tf.to_float(tf.equal(denominator, 0))
                alphas_list = tf.tile(tf.expand_dims(tf.div(e_hat_exp, denominator_regularized), 2), [1, 1, self.hidden_size])
                attentions_list = tf.multiply(alphas_list, video_embed)
                attention_embed = tf.reduce_sum(attentions_list, 0)
                #current_embed
                with tf.device("/cpu:0"):
                    current_embed_label = tf.nn.embedding_lookup(self.caption_encode_W, tf_caption_array[:,step])
                    current_embed_model = tf.nn.embedding_lookup(self.caption_encode_W, tf_max_prob_index)
                    current_embed = tf.cond(tf_sampling_choice > 0,lambda: current_embed_label, lambda: current_embed_model)
                with tf.variable_scope("LSTM1"):
                    output_1, state_1 = self.lstm_1_dropout(tf.concat([attention_embed, current_embed], 1), state_1)
                logits = tf.matmul(output_1, self.caption_decode_W) + self.caption_decode_b
                tf_max_prob_index = tf.argmax(logits, 1)
                labels = tf.sparse_to_dense(
                    tf.concat([tf.expand_dims(tf.range(self.batch_size), 1), tf.expand_dims(tf_caption_array[:, step+1], 1)], 1),
                    tf.stack([self.batch_size, self.caption_size]),
                    1.0,
                    0.0
                )
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels) * tf_caption_array_mask[:,step]
                tf_loss += tf.reduce_sum(cross_entropy)
        tf_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)     
        return tf_video_array, tf_video_array_mask, tf_caption_array, tf_caption_array_mask, tf_sampling_choice, tf_loss, tf_optimizer    
    def buildGenerator(self):
        #placeholders
        tf_video_array = tf.placeholder(tf.float32, [1, self.video_step, self.video_size])
        tf_video_array_mask = tf.placeholder(tf.float32, [1, self.video_step])
        tf_max_prob_index = tf.ones([1], dtype=tf.int32)
        tf_caption_array_id = []
        #video_encode_tensors
        video_array_flat = tf.reshape(tf_video_array, [-1, self.video_size])
        video_embed = tf.matmul(video_array_flat, self.video_encode_W) + self.video_encode_b
        video_embed = tf.reshape(video_embed, [1, self.video_step, self.hidden_size])       
        video_embed = tf.transpose(video_embed, [1, 0, 2])        
        video_embed_part = tf.reshape(video_embed, [-1, self.hidden_size])
        video_embed_part = tf.matmul(video_embed_part, self.video_attention_W) + self.video_attention_b
        video_embed_part = tf.reshape(video_embed_part, [self.video_step, 1, self.hidden_size])                
        #state_tensors
        c_init = tf.zeros([1, self.hidden_size])
        m_init = tf.zeros([1, self.hidden_size])
        #attension_tensors
        h_init = tf.zeros([1, self.hidden_size])
        with tf.variable_scope("RNN"):
            for step in range(self.caption_step-1):
                if step == 0:
                    state_1 = (c_init, m_init)
                    output_1 = h_init        			
                else:
                    tf.get_variable_scope().reuse_variables()
                #attention_embed
                e = tf.tanh(tf.matmul(output_1, self.e_attention_W) + video_embed_part)
                e = tf.reshape(e, [-1, self.hidden_size])
                e = tf.matmul(e, self.e_attention_w)
                e = tf.reshape(e, [self.video_step, 1])
                e_hat_exp = tf.multiply(tf.transpose(tf_video_array_mask), tf.exp(e))
                denominator = tf.reduce_sum(e_hat_exp, 0)
                denominator_regularized = denominator + tf.to_float(tf.equal(denominator, 0))
                alphas_list = tf.tile(tf.expand_dims(tf.div(e_hat_exp, denominator_regularized), 2), [1, 1, self.hidden_size])
                attentions_list = tf.multiply(alphas_list, video_embed)
                attention_embed = tf.reduce_sum(attentions_list, 0)
                #current_embed
                with tf.device("/cpu:0"):
                    current_embed = tf.nn.embedding_lookup(self.caption_encode_W, tf_max_prob_index)
                with tf.variable_scope("LSTM1"):
                    output_1, state_1 = self.lstm_1_dropout(tf.concat([attention_embed, current_embed], 1), state_1)
                logits = tf.matmul(output_1, self.caption_decode_W) + self.caption_decode_b
                tf_max_prob_index = tf.argmax(logits, 1)
                tf_caption_array_id.append(tf_max_prob_index)
        tf_caption_array_id = tf.stack(tf_caption_array_id)
        return tf_video_array, tf_video_array_mask, tf_caption_array_id
    