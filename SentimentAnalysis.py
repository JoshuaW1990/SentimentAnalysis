
import nltk
import nltk.corpus
from random import shuffle
from collections import defaultdict
from math import log, exp

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
        dataset = nltk.corpus.product_reviews_2
        reviews = dataset.reviews()
        features = []
        sents = []
        # Preprocess the sentence to remove the sentence with no labels
        for review in reviews:
            lines = review.review_lines
            for line in lines:
                if len(line.features) == 0:
                    continue
                features.append(line.features)
                sents.append(line.sent)
        # Divide the preprocessed_sents into two lists according to the label
        for i in range(len(features)):
            feature = features[i]
            sent_sentiment = 0
            for item in feature:
                num = int(item[1])
                sent_sentiment += num
            if sent_sentiment > 0:
                self.posSents.append(sents[i])
            elif sent_sentiment < 0:
                self.negSents.append(sents[i])
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
        # Get the minimum value of the size
        pos_size = polarity_sents.size_pos_sentence()
        neg_size = polarity_sents.size_neg_sentence()
        min_size = min(pos_size, neg_size)
        # shuffle the dataset
        pos_sentence = polarity_sents.get_pos_sentence()
        shuffle(pos_sentence)
        neg_sentence = polarity_sents.get_pos_sentence()
        shuffle(neg_sentence)

        pos_sentence = pos_sentence[:min_size]
        neg_sentence = neg_sentence[:min_size]
        return (pos_sentence, neg_sentence)

    # Divide the dataset into training set and test set by cross validation method
    def divide_dataset(self, fold_index, fold_num, polarity_sents):
        (pos_sentence, neg_sentence) = self.extract_sentence(polarity_sents)
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
        return len(self.training_input)

    def size_test_set(self):
        return len(self.test_input)

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
"""
# Multinomial naive bayes model
class Multinomial_NaiveBayesModel:
    vocabulary = set()
    total_count = 0.0
    label_count = defaultdict(float)
    pos_word_count = defaultdict(float)
    neg_word_count = defaultdict(float)
    log_prob_likelihood = defaultdict(float)

    def __init__(self, dataset = None):
        if dataset != None:
            self.train(dataset.training_input, dataset.training_output)

    def train(self, training_input, training_output):
        # Count the frequency of the label and word
        for i in range(len(training_input)):
            label = training_output[i]
            sentence = training_input[i]
            for word in sentence:
                self.vocabulary.add(word)
                self.total_count += 1.0
                self.label_count[label] += 1.0
                if label == 1:
                    self.pos_word_count[word] += 1.0
                else:
                    self.neg_word_count[word] += 1.0
        # Calculate the logaritmic value of probability
        label_set = set(training_output)
        for label in label_set:
            for word in self.vocabulary:
                item = (label, word)
                if label == 1:
                    self.log_prob_likelihood[item] = log((self.pos_word_count[word] + 1.0) / (self.label_count[label] + float(len(self.vocabulary))))
        self.check_disribution()
        return

    def check_disribution(self):
        # Check the label
        assert(self.total_count == (self.label_count[1] + self.label_count[0])), "distribution of labels are incorrect"
        # Check the word
        label_set = set(self.label_count.keys())
        for label in label_set:
            num = 0.0
            for word in self.vocabulary:
                if label == 1:
                    num += self.pos_word_count[word]
                else:
                    num += self.neg_word_count[word]
            assert(num == self.label_count[label]), "distribution of word is not correct"

    def classify(self, input):

    def test(self, test_input, test_output):

"""
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
dataset = nltk.corpus.product_reviews_1
polarityData.preprocess_dataset(dataset)
Preprocessed_dataset = ClassificationData()
Preprocessed_dataset.divide_dataset(0, 10, polarityData)

multinomial_naive_bayes_model = Multinomial_NaiveBayesModel(Preprocessed_dataset)
