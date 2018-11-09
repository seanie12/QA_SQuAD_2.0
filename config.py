class Config(object):
    def __init__(self):
        self.dropout = 0.1
        self.vocab_size = 5e4
        self.embedding_size = 300
        self.lr = 1e-3
        self.lstm_size = 128
        self.filter_size = 128
        self.attention_size = 128
        self.grad_clip = 5
        self.alpha = 1e-2
        self.beta = 1e-2
        self.l2_lambda = 3e-7
        self.num_heads = 8
        self.ans_limit = 30