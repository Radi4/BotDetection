import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm

english_stopwords = stopwords.words("english")
stemmer = SnowballStemmer("english")
tokenizer = nltk.WordPunctTokenizer()

with open('not_stem.txt', 'w') as out_1:
    with open('stem.txt', 'w') as out_2:
        with open('./data/reviewContent') as f:
            for line in tqdm(f):
                line = line.strip()
                uid, mark, year, text = line.split('\t', 3)
                text = tokenizer.tokenize(text.lower())
                text = [word for word in text if word not in string.punctuation and word not in english_stopwords]
                
                out_1.write('\t'.join([uid, mark, year, ' '.join(text)]))
                out_1.write('\n')
                
                text = list(map(stemmer.stem, text))
                out_2.write('\t'.join([uid, mark, year, ' '.join(text)]))
                out_2.write('\n')