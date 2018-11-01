from data_utils import batch_loader, Vocab, load_data, zero_padding
from model import SSNet
from config import Config
import numpy as np


vocab_file = "data/vocab"
train_file = "data/dev.txt"
max_vocab_size = 5e4
debug = True
batch_size = 16

config = Config()
vocab = Vocab(train_file, vocab_file, max_vocab_size, load=True)
q, c, s, spans, s_idx, answerable = load_data(train_file, vocab, debug)
train_data = list(zip(q, s, s_idx, answerable))
ssnet = SSNet(config)
ssnet.build_model()
for i in range(200):
    batches = batch_loader(train_data, batch_size, shuffle=True)
    for j,batch in enumerate(batches):
        batch_q, batch_s, batch_s_idx, batch_ans = zip(*batch)
        question_length, padded_q = zero_padding(batch_q, level=1)
        sequence_length, sentence_length, padded_s = zero_padding(batch_s, level=2)
        loss, acc, pred = ssnet.train(padded_q, question_length, padded_s,
                                      sequence_length, sentence_length,
                                      batch_s_idx, batch_ans)
        print(j)
        print(np.bincount(batch_ans))
        print(np.bincount(pred))
        print(loss, acc)
        print("------------------")
