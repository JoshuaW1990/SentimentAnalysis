
import nltk
import nltk.corpus
from random import shuffle
from collections import defaultdict

"""
Import Dataset
Tokenization
"""
class PolaritySents:
    def __init__(self):
        self.posSents = []
        self.negSents = []

    #example: dataset = nltk.corpus.product_reviews_2.raw()
    def preprocess_dataset(self, dataset):
        count = 1
        sents = dataset.sents()
        features = dataset.features()
        raw_txt = dataset.raw().split('\n')
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
            if sentence.startswith('##'):
                continue
            try:
                preprocessed_sents.append(sents[index])
                index += 1
            except:
                print i, index, len(raw_txt)
        # Divide the preprocessed_sents into two lists according to the label
        for i in range(len(features)):
            feature = features[i]
            sent_sentiment = 0
            for item in features[i]:
                if item == '[u]' or item == '[p]' or item == '[s]' or item == '[cs]' or item == '[cc]':
                    continue
                if not item.startswith('['):
                    continue
                num = int(item[1:-1])
                sent_sentiment += num
            if sent_sentiment > 0:
                self.posSents.append(preprocessed_sents[i])
            elif sent_sentiment < 0:
                self.negSents.append(preprocessed_sents[i])
            else:
                continue
        return

    def get_pos_sentence(self):
        return list(self.posSents)

    def size_pos_sentence(self):
        return len(self.posSents)

    def get_neg_sentence(self):
        return list(self.negSents)

    def size_neg_sentence(self):
        return len(self.negSents)



"""
Divide the dataset
We need to make sure that the number of positive sentences is equal to that of the negative sentences.
"""
class ClassificationData:
    def __init__(self):
        self.training_input = []
        self.training_output = []
        self.test_input = []
        self.test_output = []

    def extract_sentence(self, polarity_sents):
        pos_size = polarity_sents.size_pos_sentence()
        neg_size = polarity_sents.size_neg_sentence()
        if pos_size > neg_size:
            pos_sentence = polarity_sents.get_pos_sentence()
            shuffle(pos_sentence)
            pos_sentence = pos_sentence[:neg_size]
            neg_sentence = polarity_sents.get_pos_sentence()
            shuffle(neg_sentence)
        else:
            pos_sentence = polarity_sents.get_pos_sentence()
            shuffle(pos_sentence)
            neg_sentence = polarity_sents.get_pos_sentence()
            shuffle(neg_sentence)
            neg_sentence = neg_sentence[:pos_size]
        return (pos_sentence, neg_sentence)

    # Divide the dataset into training set and test set by cross validation method
    def divide_dataset(self, fold_index, fold_num, polarity_sents):
        (pos_sentence, neg_sentence) = self.divide_dataset(polarity_sents)
        if fold_index >= fold_num:
            print "error when dividing the dataset in cross validation"
            return None
        size = len(pos_sentence)
        fold_size = int(size / fold_num)
        start_index = 0
        end_index = 0
        if fold_index == fold_size - 1:
            end_index = size
        else:
            end_index = fold_size * (fold_index + 1)
        start_index = fold_size * fold_index
        self.training_input = pos_sentence[:start_index] + pos_sentence[end_index:] + neg_sentence[:start_index] + neg_sentence[end_index:]
        self.test_input = pos_sentence[start_index:end_index] + neg_sentence[start_index:end_index]
        self.training_output = [1] * (size - (end_index - start_index)) + [0] * (size - (end_index - start_index))
        self.test_output = [1] * (end_index - start_index) + [0] * (end_index - start_index)
        return None

    def size_training_set(self):
        return len(training_input)

    def size_test_set(self):
        return len(test_input)

    def get_training_input(self):
        return list(self.training_input)

    def get_training_output(self):
        return list(self.training_output)

    def get_test_input(self):
        return list(self.test_input)

    def get_test_output(self):
        return list(self.test_output)


"""
Training the dataset

# Multinomial naive bayes model
class Multinomial_NaiveBayesModel:
    def __init__(self):
        word_count = 0.0
        vocabulary_count = defaultdict(float)
        word_count = defaultdict(float)
        prob_likelihood = defaultdict(float)

    def train(self, training_input, training_output):
        for sentence in training_input:
            for word in sentence:
                word_count += 1.0



    def classify(self, input):

    def test(self, test_input, test_output):


# Bernoulli Naive bayes model
class Bernoulli_NaiveBayesModel:
    def __init__(self):



    def train(self, training_input, training_output):


    def classify(self, input):


    def test(self, test_input, test_output):
"""

polarityData = PolaritySents()
dataset = nltk.corpus.product_reviews_2
polarityData.preprocess_dataset(dataset)
#dataset = nltk.corpus.product_reviews_1
#polarityData.preprocess_dataset(dataset)
