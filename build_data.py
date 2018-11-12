from data_utils import Vocab, save_glove


def main():
    input_file = "data/train.txt"
    vocab_file = "data/vocab"
    embedding_file = "data/glove.npz"
    glove_file = "data/glove.840B.300d.txt"
    max_vocab_size = 5e4
    dim = 300
    Vocab.build_vocab(input_file, vocab_file)
    vocab = Vocab(input_file, vocab_file, max_vocab_size, load=True)
    save_glove(vocab, dim, glove_file, embedding_file)


if __name__ == "__main__":
    main()
