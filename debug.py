import nltk
import nltk.corpus

dataset = nltk.corpus.product_reviews_2
sents = dataset.sents()
features = dataset.features()
raw_txt = dataset.raw().split('\n')
preprocess_raw_txt = []
preprocessed_sents = []
print len(raw_txt)
print len(features)
print len(sents)
# Preprocess the sentence to remove the sentence with no labels
index = 0
for i in range(len(raw_txt)):
    sentence = raw_txt[i]
    if sentence.startswith('[t]') or sentence.startswith('*'):
        continue
    preprocess_raw_txt.append(sentence)
    if not sentence.startswith("##"):
        print sentence
