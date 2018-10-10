import tensorflow as tf
import numpy as np

# preprocessed data
from datasets.cornell_corpus import data
import data_utils

import importlib
importlib.reload(data)

# load data from pickle and npy files
metadata, idx_q, idx_a = data.load_data(PATH='datasets/cornell_corpus/')
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)

# parameters
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 16
xvocab_size = len(metadata['idx2w'])
yvocab_size = xvocab_size
emb_dim = 1024

import seq2seq_wrapper

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path='ckpt/cornell_corpus/',
                               emb_dim=emb_dim,
                               num_layers=3
                               )

val_batch_gen = data_utils.rand_batch_gen(validX, validY, 32)
test_batch_gen = data_utils.rand_batch_gen(testX, testY, 256)
train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, batch_size)

sess = model.restore_last_session()

input_m = test_batch_gen.__next__()

input_ = input_m[0]

print(input_m)
print(input_)


output = model.predict(sess, input_)
print(output.shape)

print(output)


replies = []
for ii, oi in zip(input_.T, output):
    # print(ii, oi)
    q = data_utils.decode(sequence=ii, lookup=metadata['idx2w'], separator=' ')
    decoded = data_utils.decode(sequence=oi, lookup=metadata['idx2w'], separator=' ').split(' ')
    # print('q : [{0}]; a : [{1}]'.format(q, ' '.join(decoded)))
    if decoded.count('unk') == 0:
        if decoded not in replies:
            print('q : [{0}]; a : [{1}]'.format(q, ' '.join(decoded)))
            replies.append(decoded)

