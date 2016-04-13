
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
    prior = defaultdict(float)
    cond_prob = defaultdict(float)
    vocabulary = set()
    total_word_count = 0.0
    label_word_count = defaultdict(float) # key: label, value:  number of words
    tuple_word_count = defaultdict(float) # key: (label, word), value: number of words

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
                self.total_word_count += 1.0
                self.label_word_count[label] += 1.0
                item = (label, word)
                self.tuple_word_count[item] += 1.0
        # Calculate the logaritmic value of probability
        label_set = set(training_output)
        for label in label_set:
            self.prior[label] = log(self.label_word_count[label] / self.total_word_count)
            for word in self.vocabulary:
                item = (label, word)
                self.cond_prob[item] = log((self.tuple_word_count[item] + 1.0) / (self.label_word_count[label] + float(len(self.vocabulary))))
        self.check_disribution()
        return

    def check_disribution(self):
        # Check the total
        assert(self.total_word_count == (self.label_word_count[1] + self.label_word_count[0])), "distribution of total is incorrect"
        # Check each label
        label_set = set(self.label_word_count.keys())
        for label in label_set:
            num = 0.0
            for word in self.vocabulary:
                item = (label, word)
                num += self.tuple_word_count[item]
            assert(num == self.label_word_count[label]), "distribution of single label is not correct"

    def classify(self, input):
        pred_output = []
        label_set = set(self.label_word_count.keys())
        for sentence in input:
            max_prob = 0.0
            pred_label = 0
            for label in range(2):
                pred_prob = self.prior[label]
                for word in sentence:
                    item = (label, word)
                    if item not in self.cond_prob:
                        continue
                    tmp_prob = self.cond_prob[item]
                    pred_prob += tmp_prob
                pred_prob = exp(pred_prob)
                print label, pred_prob, max_prob
                if pred_prob > max_prob:
                    print "reset the label"
                    max_prob = pred_prob
                    pred_label = label
            pred_output.append(pred_label)
        return pred_output

    def test(self, test_input, test_output):
        pred_output = self.classify(test_input)
        total = float(len(test_output))
        correct = 0.0
        for i in range(len(test_output)):
            if test_output[i] == pred_output[i]:
                correct += 1.0
        print correct
        print total
        accuracy = correct / total
        return accuracy

"""
# Bernoulli Naive bayes model
class Bernoulli_NaiveBayesModel:
    vocabulary = set()  # set of words in the training set
    total_sent_count = 0.0  # number of sentences in all training set
    label_sent_count = defaultdict(float) # number of sentences with a same label: key = label
    label_word_count = defaultdict(float) # number of words with a same label: key = label
    tuple_sent_count = defaultdict(float) # number of sentences containing a same word with a same label: key = (label, word)
    log_prob_likelihood = defaultdict(float) # key = (label, word)

    def __init__(self, dataset = None):
        if dataset != None:
            self.train(dataset.training_input, dataset.training_output)

    def train(self, training_input, training_output):
        for i in range(len(training_input)):
            sentence = training_input[i]
            label = training_output[i]
            self.label_sent_count[label] += 1.0
            self.total_sent_count += 1.0
            sentence_set = set()
            for word in sentence:
                self.label_word_count[label] += 1.0
                self.vocabulary.add(word)
                item = (label, word)
                if item in sentence_set:
                    continue
                sentence_set.add(item)
                self.tuple_sent_count[item] += 1.0
        for item in self.tuple_sent_count.keys():
            label = item[0]
            self.log_prob_likelihood[item] = log((self.tuple_sent_count[item] + 1.0) / (self.label_word_count[label] + 2.0))
        self.check_disribution()
        return

    def check_disribution(self):
        # Check the label
        assert(self.total_sent_count == (self.label_sent_count[1] + self.label_sent_count[0])), "distribution of labels are incorrect"


    def classify(self, input):
        pred_output = []
        label_set = set(self.label_sent_count.keys())
        max_prob = 0.0
        for sentence in input:
            for label in label_set:
                pred_label = label
                pred_prob = log(self.label_sent_count[label] / self.total_sent_count)
                sentence_set = set()
                for word in sentence:
                    item = (label, word)
                    if item not in self.log_prob_likelihood:
                        continue
                    tmp_prob = self.log_prob_likelihood[item]
                    pred_prob += tmp_prob
                if pred_prob > max_prob:
                    max_prob = pred_prob
                    pred_label = label
            pred_output.append(pred_label)



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
multinomial_accuracy = multinomial_naive_bayes_model.test(Preprocessed_dataset.get_test_input(), Preprocessed_dataset.get_test_output())
print multinomial_accuracy

"""
bernoulli_naive_bayes_model = Bernoulli_NaiveBayesModel(Preprocessed_dataset)
bernoulli_accuracy = bernoulli_naive_bayes_model.test(Preprocessed_dataset.get_test_input(), Preprocessed_dataset.get_test_output())
print bernoulli_accuracy
"""
