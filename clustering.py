import nltk
import numpy as np
import pickle
import sys
from elmoformanylangs import Embedder

model = Embedder('./models/english', batch_size = 64)
tokenizer = nltk.tokenize.WordPunctTokenizer()
sent_tokenizer = nltk.tokenize.sent_tokenize

flatten = lambda l: [item for sublist in l for item in sublist]
size = 608598
index = 0
batch = []

test_len = 0

with open('./data/reviewContent') as f:
    for line in f:
        index += 1

        uid, mark, year, sents = line.split('\t', 3)
        tokenized_text = list(map(tokenizer.tokenize, sent_tokenizer(sents.lower())))
        batch.append(tokenized_text)
        test_len += len(tokenized_text)
        if index % 1000 == 0 or index == size:
            print(test_len)
            test_len = 0
            flat_batch = flatten(batch)
            flat_len = len(flat_batch)

            emb_batch = model.sents2elmo(flat_batch)
            assert(len(emb_batch) == flat_len)

            emb_batch = np.array(list(map(lambda arr : np.mean(arr, axis = 0), emb_batch)))

            res = np.empty((len(batch), 1024))
            cumsum = 0
            for i in range(len(batch)):
                res[i] = np.mean(emb_batch[cumsum : cumsum + len(batch[i])], axis = 0)
                cumsum += len(batch[i])

            assert(cumsum == len(emb_batch))
            np.save('./embs/numpy_{}'.format(index), res)
            batch = []