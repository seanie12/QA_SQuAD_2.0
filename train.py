from data_utils import batch_loader, Vocab, load_data, zero_padding
from model import SSQANet
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
train_data = list(zip(q, c, s, s_idx, answerable, spans))
ssnet = SSQANet(config)
ssnet.build_model()
for i in range(200):
    batches = batch_loader(train_data, batch_size, shuffle=True)
    for j, batch in enumerate(batches):
        batch_q, batch_c, batch_s, batch_s_idx, batch_ans, batch_spans = zip(*batch)
        question_lengths, padded_q = zero_padding(batch_q, level=1)
        context_lengths, padded_c = zero_padding(batch_c, level=1)
        sequence_lengths, sentence_lengths, padded_s = zero_padding(batch_s, level=2)
        loss, acc, pred = ssnet.train(padded_q, question_lengths, padded_c, context_lengths,
                                      padded_s, sequence_lengths, sentence_lengths,
                                      batch_s_idx, batch_ans, batch_spans, 0.5)
        print(j)
        print(np.bincount(batch_ans))
        print(np.bincount(pred))
        print(loss, acc)
        print("------------------")
