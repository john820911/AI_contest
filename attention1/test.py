import json
import numpy as np
import tensorflow as tf
from utility import *
from parameter import *
from structure import *
from bleu import *

#argv
testing_id_file = sys.argv[1]
feature_path = sys.argv[2]

def test():
    #prepare data
    word_id, id_word, init_bias_vector, embd = loadDic(word_dic_path, id_dic_path, init_bias_dic_path, embed_dic_path)	  
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
    tf_video_array, tf_video_array_mask, tf_caption_array_id = model.buildGenerator()
    #build session, saver
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)
    session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    saver = tf.train.Saver(max_to_keep=max_to_keep)
    #restore variables   
    ckpt = tf.train.get_checkpoint_state(model_dir)
    print ("restore model from %s..." % ckpt.model_checkpoint_path)
    saver.restore(session, ckpt.model_checkpoint_path)
    #run testing
    output_list = []
    f_test_id = open(testing_id_file, "rb")
    output_file = open(output_file_path, "wb")
    for index, feat_path in enumerate(f_test_id.readlines()):
    	output_dict = {}
        video_array = np.load(test_feat_dir + feat_path.strip("\n") + ".npy")[None,...] 
        video_array_mask = np.ones((video_array.shape[0], video_array.shape[1]))
        #caption_array
        fetch_dict = {
            "caption_array_id":tf_caption_array_id
        }
        feed_dict = {
            tf_video_array:video_array,
            tf_video_array_mask:video_array_mask
        }
        track_dict = session.run(fetch_dict, feed_dict)
        caption_array_id = track_dict["caption_array_id"]
        caption_array = [ id_word[idx].encode('utf-8') for arr in caption_array_id for idx in arr ]
        output_dict["caption"] = arr2str(caption_array)
        output_dict["id"] = feat_path.strip("\n")
        output_list.append(output_dict)
    json.dump(output_list, output_file, indent=4)
    output_file.close()

if __name__ == "__main__":
    test()
