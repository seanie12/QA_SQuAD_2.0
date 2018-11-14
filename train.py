from data_utils import batch_loader, Vocab, load_data, zero_padding
from model import SSQANet
from config import Config
import numpy as np


def main():
    config = Config()
    vocab = Vocab(config.dict_file)
    q, c, s, spans, s_idx, answerable = load_data(config.train_file, vocab, config.debug)
    dev_q, dev_c, dev_s, dev_spans, dev_s_idx, dev_answerable = load_data(config.dev_file, vocab, config.debug)
    train_data = list(zip(q, c, s, s_idx, answerable, spans))
    dev_data = list(zip(dev_q, dev_c, dev_s, dev_s_idx, dev_answerable, dev_spans))
    ssnet = SSQANet(config)
    ssnet.build_model()
    best_score = 0
    for i in range(config.num_epochs):
        epoch = i + 1
        batches = batch_loader(train_data, config.batch_size, shuffle=False)
        for batch in batches:
            batch_q, batch_c, batch_s, batch_s_idx, batch_ans, batch_spans = zip(*batch)
            question_lengths, padded_q = zero_padding(batch_q, level=1)
            context_lengths, padded_c = zero_padding(batch_c, level=1)
            sequence_lengths, sentence_lengths, padded_s = zero_padding(batch_s, level=2)
            loss, acc, pred, step = ssnet.train(padded_q, question_lengths, padded_c, context_lengths,
                                                padded_s, sequence_lengths, sentence_lengths,
                                                batch_s_idx, batch_ans, batch_spans, config.dropout)
            train_batch_acc, train_batch_em, train_batch_loss = ssnet.eval(padded_q, question_lengths, padded_c,
                                                                           context_lengths,
                                                                           padded_s, sequence_lengths,
                                                                           sentence_lengths, batch_s_idx,
                                                                           batch_ans, batch_spans)
            if step % 100 == 0:
                print("epoch: %d, step:%d, loss:%.4f, acc:%.2f, em:%.2f"
                      % (epoch, step, loss, train_batch_acc, train_batch_em))

            if step % 1000 == 0:
                dev_batches = batch_loader(dev_data, config.batch_size, shuffle=False)
                total_em = []
                total_acc = []
                total_loss = []
                for dev_batch in dev_batches:
                    dev_batch_q, dev_batch_c, dev_batch_s, \
                    dev_batch_s_idx, dev_batch_ans, dev_batch_spans = zip(*dev_batch)
                    question_lengths, padded_q = zero_padding(dev_batch_q, level=1)
                    context_lengths, padded_c = zero_padding(dev_batch_c, level=1)
                    sequence_lengths, sentence_lengths, padded_s = zero_padding(dev_batch_s, level=2)
                    dev_batch_acc, dev_batch_em, dev_batch_loss = ssnet.eval(padded_q, question_lengths, padded_c,
                                                                             context_lengths,
                                                                             padded_s, sequence_lengths,
                                                                             sentence_lengths, dev_batch_s_idx,
                                                                             dev_batch_ans, dev_batch_spans)

                    total_loss.append(dev_batch_loss)
                    total_em.append(dev_batch_em)
                    total_acc.append(dev_batch_acc)
                dev_em = np.mean(total_em)
                dev_acc = np.mean(total_acc)
                dev_loss = np.mean(total_loss)
                ssnet.write_summary(dev_acc, dev_em, dev_loss, mode="dev")
                ssnet.write_summary(train_batch_acc, train_batch_em, train_batch_loss, mode="train")
                print("after %d step, dev_em:%.2f" % (step, dev_em))
                if dev_em > best_score:
                    best_score = dev_em
                    print("new score! em: %.2f, acc:%.2f" % (dev_em, dev_acc))
                    ssnet.save_session(config.dir_model)


if __name__ == "__main__":
    main()
