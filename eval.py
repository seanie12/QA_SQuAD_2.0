from data_utils import batch_loader, Vocab, load_data, zero_padding
from model import SSQANet
from config import Config
import numpy as np


def main():
    config = Config()
    vocab = Vocab(config.dict_file)
    dev_q, dev_c, dev_s, dev_spans, dev_s_idx, dev_answerable = load_data(config.dev_file, vocab, config.debug)
    dev_data = list(zip(dev_q, dev_c, dev_s, dev_s_idx, dev_answerable, dev_spans))
    ssnet = SSQANet(config)
    ssnet.build_model()
    ssnet.restore_session(config.dir_model)
    batches = batch_loader(dev_data, config.batch_size, shuffle=False)
    acc_history = []
    em_history = []
    for batch in batches:
        batch_q, batch_c, batch_s, batch_s_idx, batch_ans, batch_spans = zip(*batch)
        question_lengths, padded_q = zero_padding(batch_q, level=1)
        context_lengths, padded_c = zero_padding(batch_c, level=1)
        sequence_lengths, sentence_lengths, padded_s = zero_padding(batch_s, level=2)

        batch_acc, batch_em, batch_loss = ssnet.eval(padded_q, question_lengths, padded_c, context_lengths,
                                                     padded_s, sequence_lengths, sentence_lengths, batch_s_idx,
                                                     batch_ans, batch_spans)
        acc_history.append(batch_acc)
        em_history.append(batch_em)

    dev_acc = np.mean(acc_history)
    dev_em = np.mean(em_history)
    print("classification acc :{}".format(dev_acc))
    print("EM :{}".format(dev_em))


if __name__ == "__main__":
    main()
