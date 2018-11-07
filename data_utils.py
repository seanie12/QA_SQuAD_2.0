import numpy as np
import random

PAD = "<PAD>"
DUMMY = "<d>"
UNK = "<UNK>"
ID_PAD = 0
ID_DUMMY = 1
ID_UNK = 2
special_tokens = [PAD, DUMMY, UNK]


class Vocab(object):
    def __init__(self, input_file, vocab_file, max_vocab_size, load=False):
        # input file is tab delimiter question / context / answer/...
        # we need question and context
        self._word2idx = dict()
        # read questions and context, and count frequency of each token
        if load:
            for token in special_tokens:
                idx = len(self._word2idx)
                self._word2idx[token] = idx
            with open(vocab_file, "r", encoding="utf-8") as f:
                count = len(special_tokens)
                for line in f:
                    word, freq = line.split("\t")
                    if word not in self._word2idx:
                        idx = len(self._word2idx)
                        self._word2idx[word] = idx
                        count += 1
                        if count >= max_vocab_size:
                            break
        else:
            Vocab.build_vocab(input_file, vocab_file)

    @staticmethod
    def build_vocab(input_file, vocab_file):
        counter = dict()
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                contents = line.split("\t")
                question = contents[0].strip()
                context = contents[1].strip()
                for q_token in question.split():
                    q_token = q_token.strip()
                    if q_token in counter:
                        counter[q_token] += 1
                    else:
                        counter[q_token] = 1
                for c_token in context.split():
                    c_token = c_token.strip()
                    if c_token in special_tokens or c_token == "</s>":
                        continue
                    if c_token in counter:
                        counter[c_token] += 1
                    else:
                        counter[c_token] = 1
        # sort frequency by descending order
        sorted_counter = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)
        with open(vocab_file, "w", encoding="utf-8") as f:
            for word, freq in sorted_counter:
                f.write(word + "\t" + str(freq) + "\n")

    def word2idx(self, word):
        if word in self._word2idx:
            return self._word2idx[word]
        else:
            return self._word2idx[UNK]


def batch_loader(iterable, batch_size, shuffle=False):
    length = len(iterable)
    # shuffle iterable
    if shuffle:
        random.shuffle(iterable)
    for start_idx in range(0, length, batch_size):
        yield iterable[start_idx:min(length, start_idx + batch_size)]


def load_data(input_file, vocab, debug=False):
    with open(input_file, "r", encoding="utf-8") as f:
        questions = []
        contexts = []
        sentences = []
        spans = []
        sentence_idx = []
        answerable = []
        for i, line in enumerate(f):
            # debug mode
            if debug and i == 100:
                break
            # question / context / start_idx / end_idx / oracle_sentence_idx / answerable
            contents = line.split("\t")
            q = contents[0]
            context = contents[1]
            # append answer spans, sentence_idx, and answerable
            spans.append([int(contents[2]), int(contents[3])])
            sentence_idx.append(int(contents[4]))
            answerable.append(int(contents[5]))
            # tokenize question and append it to list
            q_tokens = q.split()
            # map token to idx
            q_tok2idx = list(map(lambda token: vocab.word2idx(token), q_tokens))
            questions.append(q_tok2idx)
            # remove sentence delimeter, tokenize and append it to list
            c_tokens = context.replace("</s>", "").split()
            # map token to its idx
            c_tokens = list(map(lambda token: vocab.word2idx(token), c_tokens))
            contexts.append(c_tokens)
            # split paragraphs into sentences
            c_sentences = context.split("</s>")
            # tokenize each sentence, map token to idx and append it to list
            # [[word1, word2, word3..], [word4, word5..]]
            c_tokenized_sentences = list(
                map(lambda sentence: [vocab.word2idx(word) for word in sentence.split()], c_sentences))
            sentences.append(c_tokenized_sentences)
        return questions, contexts, sentences, spans, sentence_idx, answerable


def zero_padding(inputs, level):
    # inputs : [batch, sentence_len] or [batch, doc_len, sentence_len]
    if level == 1:
        sequence_length = [len(doc) for doc in inputs]
        max_length = max(sequence_length)
        padded_docs = list(map(lambda doc: doc + [ID_PAD] * (max_length - len(doc)), inputs))
        return sequence_length, padded_docs

    elif level == 2:
        max_doc_len = max([len(doc) for doc in inputs])
        max_sentence_len = max([max(map(lambda x: len(x), doc)) for doc in inputs])
        padded_docs = []
        sequence_length = []
        doc_lengths = []
        for doc in inputs:
            # calculate the number of sentences in a document
            doc_length = len(doc)
            doc_lengths.append(doc_length)
            # calculate the number of words in each sentence
            sentence_lengths = list(map(lambda sentence: len(sentence), doc))
            # add the number of zero padding sentence
            sentence_lengths += [ID_PAD] * (max_doc_len - doc_length)
            sequence_length.append(sentence_lengths)
            # make all sentences as the same length
            padded_doc = list(map(lambda sentence: sentence + [ID_PAD] * (max_sentence_len - len(sentence)), doc))
            # make a document with all the same number of sentences
            padded_doc += [[ID_PAD] * max_sentence_len] * (max_doc_len - doc_length)
            padded_docs.append(padded_doc)

        return np.array(sequence_length), np.array(doc_lengths), np.array(padded_docs)


if __name__ == "__main__":
    vocab = Vocab("data/train.txt", "data/vocab", 5e4, load=True)
    docs = [
        [[1, 2, 3, ], [4, 5, 6, 7, ]],
        [[5, 7], [8, 9, 11], [1], [5]],
        [[1]]
    ]
    q, c, sentences, span, sent_idx, answerable = load_data("data/train.txt", vocab, debug=True)
    sequence_length_, doc_lengths_, padded_docs_ = zero_padding(sentences, level=2)
    print(sequence_length_, sequence_length_.shape)
    print(doc_lengths_, doc_lengths_.shape)
    print(padded_docs_)