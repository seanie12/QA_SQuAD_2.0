from data_utils import Vocab


def main():
    input_file = "data/train.txt"
    vocab_file = "data/vocab"
    embedding_file = "data/glove.npz"
    glove_file = "data/glove.840B.300d.txt"
    dict_file = "data/dict.p"
    max_vocab_size = 5e4
    Vocab.build_vocab(input_file, vocab_file, dict_file, glove_file, embedding_file, max_vocab_size)


if __name__ == "__main__":
    main()
