class Config(object):
    def __init__(self):
        self.dropout = 0.5
        self.vocab_size = 5e4
        self.embedding_size = 300
        self.lr = 1e-3
        self.lstm_size = 128
        self.attention_size = 128
        self.clip = 5
        self.alpha = 1e-2

