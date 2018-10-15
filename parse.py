import json
from nltk.tokenize import sent_tokenize, word_tokenize

input_file = "data/train-v2.0.json"
output_file = "data/train.txt"
DUMMY = "<d>"


# define tokenize by nltk
def tokenize(_context, sentence_split=False, append_dummy=False):
    sentContext = sent_tokenize(_context)
    words = []
    for sent in sentContext:
        wordContext = word_tokenize(sent)
        for word in wordContext:
            words.append(word)
        # add sentence split delimeter to the end of sentence
        # except the last sentence
        if sentence_split:
            words.append("</s>")
    if append_dummy:
        words.append(DUMMY)
    return words


def select_oracle_sentence(sentences, answer_span):
    # return idx of sentence which contains answer span
    # if there is no such sentence, return num_sentence
    # because the last sentence would be a dummy sentence
    idx = len(sentences)
    for i, sentence in enumerate(sentences):
        if answer_span in sentence:
            idx = i
    return idx


# load json file
dict_ = json.load(open(input_file))

contextIndex = -1
answerable = None
# parse the json file and write the result into text file
fw = open(output_file, "w", encoding="utf-8")
titleNum = len(dict_['data'])
for i in range(titleNum):
    title = dict_['data'][i]['title']
    paragraphNum = len(dict_['data'][i]['paragraphs'])
    for j in range(paragraphNum):
        contextIndex += 1
        context = dict_['data'][i]['paragraphs'][j]['context']
        context_with_delimeter = tokenize(context, sentence_split=True, append_dummy=True)
        context_join = " ".join(context_with_delimeter).lower()
        qasNum = len(dict_['data'][i]['paragraphs'][j]['qas'])
        for k in range(qasNum):
            question = dict_['data'][i]['paragraphs'][j]['qas'][k]['question']
            question = tokenize(question)
            question = " ".join(question).lower()
            answersList = dict_['data'][i]['paragraphs'][j]['qas'][k]['answers']
            sentences = sent_tokenize(context)
            sentence_idx = -1
            if len(answersList) == 0:
                # if a question is unanswerable, answer idx is the
                # index of dummy word, which is the last word appended to end of document
                answerable = 0
                answerText = ''
                startPosition = len(tokenize(context))
                endPosition = startPosition
                sentence_idx = len(sentences)
            else:
                # if there are multiple answers, we chose the first answer
                answerable = 1
                answerText = dict_['data'][i]['paragraphs'][j]['qas'][k]['answers'][0]['text']
                charPosition = dict_['data'][i]['paragraphs'][j]['qas'][k]['answers'][0]['answer_start']
                startPosition = len(tokenize(context[:charPosition]))
                tokenizedAnswer = tokenize(answerText)
                endPosition = startPosition + len(tokenizedAnswer) - 1
                sentence_idx = select_oracle_sentence(sentences, answerText)
            fw.write(question + "\t" + context_join + "\t" + str(startPosition) + "\t"
                     + str(endPosition) + "\t" + str(sentence_idx) + "\t" + str(answerable) + "\n")

fw.close()
