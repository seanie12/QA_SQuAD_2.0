from data_utils import load_glove
import os


class Config(object):
    def __init__(self):
        self.vocab_file = "data/vocab"
        self.train_file = "data/train.txt"
        self.dev_file = "data/dev.txt"
        self.dict_file = "data/dict.p"
        self.max_vocab_size = 5e4
        self.debug = True
        self.num_epochs = 20
        self.batch_size = 16
        self.dropout = 0.1
        self.vocab_size = 5e4
        self.embedding_size = 300
        self.lr = 1e-3
        self.lstm_size = 128
        self.filter_size = 128
        self.attention_size = 128
        self.grad_clip = 5
        self.alpha = 1e-1
        self.beta = 1e-1
        self.l2_lambda = 3e-7
        self.num_heads = 8
        self.ans_limit = 20
        self.embeddings = load_glove("data/glove.npz")
        self.dir_output = "results/save/"
        self.dir_model = self.dir_output + "model.weights/"
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)
