import os
import sys
import time
import json
import numpy as np
import tensorflow as tf
from utility import *
from parameter import *
from structure import *

def train():
    #prepare data
    train_data, test_data = getInfo(train_info_path), getInfo(test_info_path)
    word_id, _, init_bias_vector, embd = loadDic(word_dic_path, id_dic_path, init_bias_dic_path, embed_dic_path)
    #initialize model
    model = VideoCaptionGenerator(
            video_size=video_size,
            video_step=video_step,
            caption_size=caption_size,
            caption_step=caption_step,
            hidden_size=hidden_size,
            batch_size=batch_size,
            output_keep_prob=output_keep_prob,
            init_bias_vector=init_bias_vector,
    		pretrained_embd=embd
        )
    #build model
    tf_video_array, tf_video_array_mask, tf_caption_array, tf_caption_array_mask, tf_sampling_choice, tf_loss, tf_optimizer = model.buildModel()
    #build session, saver
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)
    session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    saver = tf.train.Saver(max_to_keep=max_to_keep)
    if restore_flag == True:
        saver.restore(session, train_model_path)
    #initialize variables
    else:
        tf.global_variables_initializer().run()
    #run epochs
    for epoch in range(num_epoch):
        #shuffle
        if epoch <= train_model_version:
            continue		
        index_list = np.arange(len(train_data))
        np.random.shuffle(index_list)
        current_train_data = train_data.ix[index_list]
        #batch
        start_time = time.time()        
        for start, end in zip(range(0, len(current_train_data), batch_size), range(batch_size, len(current_train_data), batch_size)):
            #video, caption batch
            current_batch = current_train_data[start:end]
            current_video_batch = map(lambda x: np.load(train_feat_dir + x), current_batch["feat_path"].values)
            current_caption_batch = [ "<bos> " + json.load(open(train_label_dir + path)) + " <eos>" for path in current_batch["label_path"].values ]
            current_caption_id_batch = [ [ word_id[word] for word in sentence.lower().split(" ") if word in word_id ] for sentence in current_caption_batch ]
            #video_array
            video_array = np.zeros((batch_size, video_step, video_size), dtype="float32")
            for index, video in enumerate(current_video_batch):
                video_array[index] = video
            #video_array_mask
            video_array_mask = np.ones((batch_size, video_step))
            #caption_array
            caption_array = np.zeros((batch_size, caption_step), dtype="int32")
            for index in range(len(current_caption_id_batch)):
                caption_array[index, :len(current_caption_id_batch[index])] = current_caption_id_batch[index]
            #caption_array_mask
            caption_array_mask = np.zeros((batch_size, caption_step))
            nonzero_length = np.array(map(lambda x: (x != 0).sum() - 1, caption_array))
            for index, row in enumerate(caption_array_mask):
                row[:nonzero_length[index]] = 1
            #loss
            fetch_dict = {
                "loss":tf_loss,
                "optimizer":tf_optimizer,
            }
            # schedule sampling
            sampling_prob = 1 - float(epoch)/1000 # linear
            # sampling_prob = (1- 23e-4) ** epoch # exponential
            # sampling_prob = inv_sigmoid(epoch, 215) # inverse sigmoid
            s = np.random.binomial(1, sampling_prob)
            # print float(sum(s)) / 10000, sampling_prob, epoch
            sampling_choice = np.array(s, dtype="int32")
            feed_dict = {
                tf_video_array:video_array,
                tf_video_array_mask:video_array_mask,
                tf_caption_array:caption_array,
                tf_caption_array_mask:caption_array_mask,
                tf_sampling_choice:sampling_choice
            }
            track_dict = session.run(fetch_dict, feed_dict)
            #print
            sys.stdout.write("\rBatchID: {0}, Loss: {1}".format(start / batch_size, track_dict["loss"]))
            sys.stdout.flush()
        end_time = time.time()
        sys.stdout.write("\nEpoch: {0}, Loss: {1}, Time: {2}\n".format(epoch, track_dict["loss"], end_time - start_time))
        #save
        if np.mod(epoch, save_per_epoch) == 0:
            print "Epoch ", epoch, " is done. Saving the model..."
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            saver.save(session, model_dir, global_step=epoch)            

if __name__ == "__main__":
    train()
