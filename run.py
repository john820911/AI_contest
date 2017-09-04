from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import math
import random
import numpy as np
import tensorflow as tf
from six.moves import xrange
import utility
import model

try:
    reload
except NameError:
    pass
else:
    reload(sys).setdefaultencoding("utf-8")
    
try:
    from ConfigParser import SafeConfigParser
except:
    from configparser import SafeConfigParser
    
gConfig = {}
_buckets = [(5, 5), (8, 8), (15, 15), (30, 30)]

def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def get_config(config_file="info"):
    parser = SafeConfigParser()
    parser.read(config_file)
    _conf_ints = [ (key, int(value)) for key,value in parser.items("ints") ]
    _conf_floats = [ (key, float(value)) for key,value in parser.items("floats") ]
    _conf_strings = [ (key, str(value)) for key,value in parser.items("strings") ]
    return dict(_conf_ints + _conf_floats + _conf_strings)

def read_data(source_path, target_path, max_size=None):
    data_set, data_set_ordered = [ [] for _ in _buckets ], []
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [ int(x) for x in source.split() ]
                target_ids = [ int(x) for x in target.split() ]
                target_ids.append(utility.EOS_ID)
                data_set_ordered.append([source_ids, target_ids])
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set, data_set_ordered

def create_model(session, forward_only):
    attenSeq2Seq = model.attenSeq2Seq(
        gConfig["enc_vocab_size"],
        gConfig["dec_vocab_size"],
        _buckets,
        gConfig["layer_size"],
        gConfig["num_layers"],
        gConfig["max_gradient_norm"],
        gConfig["batch_size"],
        gConfig["learning_rate"],
        gConfig["learning_rate_decay_factor"],
        max_to_keep=gConfig["max_to_keep"],
        forward_only=forward_only
    )
    ckpt = tf.train.get_checkpoint_state(gConfig["model_directory"])
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path + ".index"):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        attenSeq2Seq.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return attenSeq2Seq

def train():
    print("Preparing data in %s" % gConfig["dict_directory"])
    enc_train, dec_train, enc_dev, dec_dev, _, _ = utility.prepare_custom_data(gConfig["dict_directory"], gConfig["train_enc"], gConfig["train_dec"], gConfig["dev_enc"], gConfig["dev_dec"],gConfig["enc_vocab_size"],gConfig["dec_vocab_size"])
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.666)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allocator_type = "BFC"
    with tf.Session(config=config) as sess:
        print("Creating %d layers of %d units." % (gConfig["num_layers"], gConfig["layer_size"]))
        model = create_model(sess, False)
        print ("Reading development and training data (limit: %d)." % gConfig['max_train_data_size'])
        dev_set, _ = read_data(enc_dev, dec_dev)
        train_set, _ = read_data(enc_train, dec_train, gConfig["max_train_data_size"])
        train_bucket_sizes = [ len(train_set[b]) for b in xrange(len(_buckets)) ]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [ sum(train_bucket_sizes[:i + 1]) / train_total_size for i in xrange(len(train_bucket_sizes)) ]
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([ i for i in xrange(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01 ])
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)
            _, step_loss, _ = model.step(
                sess,
                encoder_inputs,
                decoder_inputs,
                target_weights,
                bucket_id,
                False
            )
            step_time += (time.time() - start_time) / gConfig["steps_per_checkpoint"]
            loss += step_loss / gConfig["steps_per_checkpoint"]
            current_step += 1
            if current_step % gConfig["steps_per_checkpoint"] == 0:
                perplexity = math.exp(loss) if loss < 300 else float("inf")
                print ("global step %d learning rate %.4f step-time %.2f perplexity %.2f" % 
                        (model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity))
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                checkpoint_path = os.path.join(gConfig["model_directory"], "attenSeq2Seq.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                for bucket_id in xrange(len(_buckets)):
                    if len(dev_set[bucket_id]) == 0:
                        print("  eval: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(dev_set, bucket_id)
                    _, eval_loss, _ = model.step(
                        sess,
                        encoder_inputs,
                        decoder_inputs,
                        target_weights,
                        bucket_id,
                        True
                    )
                    eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float("inf")
                    print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                sys.stdout.flush()

def decode():
    print("Preparing data in %s" % gConfig["dict_directory"])
    _, _, enc_dev, dec_dev, _, _ = utility.prepare_custom_data(gConfig["dict_directory"], gConfig["train_enc"], gConfig["train_dec"], gConfig["dev_enc"], gConfig["dev_dec"],gConfig["enc_vocab_size"],gConfig["dec_vocab_size"])
    _, _, enc_test, dec_test_0, _, _ = utility.prepare_custom_data(gConfig["dict_directory"], gConfig["train_enc"], gConfig["train_dec"], gConfig["test_enc"], gConfig["test_0_dec"],gConfig["enc_vocab_size"],gConfig["dec_vocab_size"])
    _, _, enc_test, dec_test_1, _, _ = utility.prepare_custom_data(gConfig["dict_directory"], gConfig["train_enc"], gConfig["train_dec"], gConfig["test_enc"], gConfig["test_1_dec"],gConfig["enc_vocab_size"],gConfig["dec_vocab_size"])
    _, _, enc_test, dec_test_2, _, _ = utility.prepare_custom_data(gConfig["dict_directory"], gConfig["train_enc"], gConfig["train_dec"], gConfig["test_enc"], gConfig["test_2_dec"],gConfig["enc_vocab_size"],gConfig["dec_vocab_size"])
    _, _, enc_test, dec_test_3, _, _ = utility.prepare_custom_data(gConfig["dict_directory"], gConfig["train_enc"], gConfig["train_dec"], gConfig["test_enc"], gConfig["test_3_dec"],gConfig["enc_vocab_size"],gConfig["dec_vocab_size"])
    _, _, enc_test, dec_test_4, _, _ = utility.prepare_custom_data(gConfig["dict_directory"], gConfig["train_enc"], gConfig["train_dec"], gConfig["test_enc"], gConfig["test_4_dec"],gConfig["enc_vocab_size"],gConfig["dec_vocab_size"])
    _, _, enc_test, dec_test_5, _, _ = utility.prepare_custom_data(gConfig["dict_directory"], gConfig["train_enc"], gConfig["train_dec"], gConfig["test_enc"], gConfig["test_5_dec"],gConfig["enc_vocab_size"],gConfig["dec_vocab_size"])    
    file = open(gConfig["result"], "w")
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        print("Creating %d layers of %d units." % (gConfig["num_layers"], gConfig["layer_size"]))
        model = create_model(sess, True)
        print ("Reading development and training data (limit: %d)." % gConfig['max_train_data_size'])
        dev_set_0, dev_set_0_ordered = read_data(enc_dev, dec_dev)
        test_set_0, test_set_0_ordered = read_data(enc_test, dec_test_0)
        test_set_1, test_set_1_ordered = read_data(enc_test, dec_test_1)
        test_set_2, test_set_2_ordered = read_data(enc_test, dec_test_2)
        test_set_3, test_set_3_ordered  = read_data(enc_test, dec_test_3)
        test_set_4, test_set_4_ordered  = read_data(enc_test, dec_test_4)
        test_set_5, test_set_5_ordered  = read_data(enc_test, dec_test_5)
        model.batch_size = 1
        enc_vocab_path = os.path.join(gConfig["dict_directory"], "vocab%d.enc" % gConfig["enc_vocab_size"])
        dec_vocab_path = os.path.join(gConfig["dict_directory"], "vocab%d.dec" % gConfig["dec_vocab_size"])
        enc_vocab, _ = utility.initialize_vocabulary(enc_vocab_path)
        _, rev_dec_vocab = utility.initialize_vocabulary(dec_vocab_path)
        result_list = list()
        for test_set in test_set_0_ordered:
            test_bucket_id = 0
            for bucket_id in range(len(_buckets)):
                found = False
                for token_ids_pair in test_set_0[bucket_id]:
                    if test_set[0] == token_ids_pair[0]:
                        test_bucket_id = bucket_id
                        found = True
                        break
                if found == True:
                    break
            encoder_inputs, decoder_inputs, target_weights = model.get_batch({ test_bucket_id: [ (test_set[0], test_set[1]) ] }, test_bucket_id)
            _, eval_loss, output_logits = model.step(
                sess,
                encoder_inputs,
                decoder_inputs,
                target_weights,
                test_bucket_id,
                True
            )           
            result_list.append([ math.exp(eval_loss) ])
        print("Finish test_0")
        for i, test_set in enumerate(test_set_1_ordered):
            test_bucket_id = 0
            for bucket_id in range(len(_buckets)):
                found = False
                for token_ids_pair in test_set_1[bucket_id]:
                    if test_set[0] == token_ids_pair[0]:
                        test_bucket_id = bucket_id
                        found = True
                        break
                if found == True:
                    break
            encoder_inputs, decoder_inputs, target_weights = model.get_batch({ test_bucket_id: [ (test_set[0], test_set[1]) ] }, test_bucket_id)
            _, eval_loss, output_logits = model.step(
                sess,
                encoder_inputs,
                decoder_inputs,
                target_weights,
                test_bucket_id,
                True
            )         
            result_list[i].append(math.exp(eval_loss))       
        print("Finish test_1")
        for i, test_set in enumerate(test_set_2_ordered):
            test_bucket_id = 0
            for bucket_id in range(len(_buckets)):
                found = False
                for token_ids_pair in test_set_2[bucket_id]:
                    if test_set[0] == token_ids_pair[0]:
                        test_bucket_id = bucket_id
                        found = True
                        break
                if found == True:
                    break
            encoder_inputs, decoder_inputs, target_weights = model.get_batch({ test_bucket_id: [ (test_set[0], test_set[1]) ] }, test_bucket_id)
            _, eval_loss, output_logits = model.step(
                sess,
                encoder_inputs,
                decoder_inputs,
                target_weights,
                test_bucket_id,
                True
            )            
            result_list[i].append(math.exp(eval_loss))
        print("Finish test_2")
        for i, test_set in enumerate(test_set_3_ordered):
            test_bucket_id = 0
            for bucket_id in range(len(_buckets)):
                found = False
                for token_ids_pair in test_set_3[bucket_id]:
                    if test_set[0] == token_ids_pair[0]:
                        test_bucket_id = bucket_id
                        found = True
                        break
                if found == True:
                    break
            encoder_inputs, decoder_inputs, target_weights = model.get_batch({ test_bucket_id: [ (test_set[0], test_set[1]) ] }, test_bucket_id)
            _, eval_loss, output_logits = model.step(
                sess,
                encoder_inputs,
                decoder_inputs,
                target_weights,
                test_bucket_id,
                True
            )            
            result_list[i].append(math.exp(eval_loss))
        print("Finish test_3")
        for i, test_set in enumerate(test_set_4_ordered):
            test_bucket_id = 0
            for bucket_id in range(len(_buckets)):
                found = False
                for token_ids_pair in test_set_4[bucket_id]:
                    if test_set[0] == token_ids_pair[0]:
                        test_bucket_id = bucket_id
                        found = True
                        break
                if found == True:
                    break
            encoder_inputs, decoder_inputs, target_weights = model.get_batch({ test_bucket_id: [ (test_set[0], test_set[1]) ] }, test_bucket_id)
            _, eval_loss, output_logits = model.step(
                sess,
                encoder_inputs,
                decoder_inputs,
                target_weights,
                test_bucket_id,
                True
            )           
            result_list[i].append(math.exp(eval_loss))
        print("Finish test_4")
        for i, test_set in enumerate(test_set_5_ordered):
            test_bucket_id = 0
            for bucket_id in range(len(_buckets)):
                found = False
                for token_ids_pair in test_set_5[bucket_id]:
                    if test_set[0] == token_ids_pair[0]:
                        test_bucket_id = bucket_id
                        found = True
                        break
                if found == True:
                    break
            encoder_inputs, decoder_inputs, target_weights = model.get_batch({ test_bucket_id: [ (test_set[0], test_set[1]) ] }, test_bucket_id)
            _, eval_loss, output_logits = model.step(
                sess,
                encoder_inputs,
                decoder_inputs,
                target_weights,
                test_bucket_id,
                True
            )          
            result_list[i].append(math.exp(eval_loss))
        print("Finish test_5")      
        for result in result_list:
            file.write(str(np.argmin(result)) + "\n")

# def decode():
#   # Only allocate part of the gpu memory when predicting.
#   gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
#   config = tf.ConfigProto(gpu_options=gpu_options)

#   with tf.Session(config=config) as sess:
#     # Create model and load parameters.
#     model = create_model(sess, True)
#     model.batch_size = 1  # We decode one sentence at a time.

#     # Load vocabularies.
#     enc_vocab_path = os.path.join(gConfig['dict_directory'],"vocab%d.enc" % gConfig['enc_vocab_size'])
#     dec_vocab_path = os.path.join(gConfig['dict_directory'],"vocab%d.dec" % gConfig['dec_vocab_size'])
#     enc_vocab, _ = utility.initialize_vocabulary(enc_vocab_path)
#     _, rev_dec_vocab = utility.initialize_vocabulary(dec_vocab_path)

#     # Decode from standard input.
#     sys.stdout.write("> ")
#     sys.stdout.flush()
#     sentence = sys.stdin.readline()
#     while sentence:
#       # Get token-ids for the input sentence.
#       token_ids = utility.sentence_to_token_ids(tf.compat.as_bytes(sentence), enc_vocab)
#       # Which bucket does it belong to?
#       bucket_id = min([b for b in xrange(len(_buckets))
#                        if _buckets[b][0] > len(token_ids)])
#       # Get a 1-element batch to feed the sentence to the model.
#       encoder_inputs, decoder_inputs, target_weights = model.get_batch(
#           {bucket_id: [(token_ids, [])]}, bucket_id)
#       # Get output logits for the sentence.
#       _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
#                                        target_weights, bucket_id, True)
#       # This is a greedy decoder - outputs are just argmaxes of output_logits.
#       outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
#       # If there is an EOS symbol in outputs, cut them at that point.
#       if utility.EOS_ID in outputs:
#         outputs = outputs[:outputs.index(utility.EOS_ID)]
#       # Print out French sentence corresponding to outputs.
#       print(" ".join([tf.compat.as_str(rev_dec_vocab[output]) for output in outputs]))
#       print("> ", end="")
#       sys.stdout.flush()
#       sentence = sys.stdin.readline()

if __name__ == "__main__":
    gConfig = get_config()
    print("\n>> Mode : %s\n" %(gConfig["mode"]))
    if gConfig["mode"] == "train":
        train()
    elif gConfig["mode"] == "test":
        decode()
    else:
        print("# uses seq2seq_serve.ini as conf file")
