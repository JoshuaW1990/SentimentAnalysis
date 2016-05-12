import nltk
import nltk.corpus
from nltk.classify import MaxentClassifier
import numpy as np
from random import shuffle
from collections import defaultdict
from math import log, exp, sqrt



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

"""
Preprocess the dataset:
Divde the dataset into training set and test set
Find the best feature from the vocabulary of the dataset
label for positive sentence: 1
label for negative sentence: 0
"""
class Preprocess_Data:
    def __init__(self):
        self.X_train =[]
        self.Y_train = []
        self.X_test = []
        self.Y_test = []
        self.word_features = set()

    def extract_sentence(self, polarity_sents):
        # Get the minimum value of the size
        pos_size = len(polarity_sents.posSents)
        neg_size = len(polarity_sents.negSents)
        min_size = min(pos_size, neg_size)
        # shuffle the dataset
        pos_sentence = polarity_sents.posSents
        shuffle(pos_sentence)
        neg_sentence = polarity_sents.negSents
        shuffle(neg_sentence)

        pos_sentence = pos_sentence[:min_size]
        neg_sentence = neg_sentence[:min_size]
        return (pos_sentence, neg_sentence)

    # Divide the dataset into training set and test set by cross validation method
    # Divide the dataset into training set and test set by cross validation method
    def divide_dataset(self, fold_index, fold_num, pos_sentence, neg_sentence):
        if fold_index >= fold_num:
            print "error when dividing the dataset in cross validation"
            return None
        size = len(pos_sentence)
        fold_size = int(size / fold_num)
        if fold_index == fold_size - 1:
            end_index = size
        else:
            end_index = fold_size * (fold_index + 1)
        start_index = fold_size * fold_index
        training_input = pos_sentence[:start_index] + pos_sentence[end_index:] + neg_sentence[:start_index] + neg_sentence[end_index:]
        test_input = pos_sentence[start_index:end_index] + neg_sentence[start_index:end_index]
        training_output = [1] * (size - (end_index - start_index)) + [-1] * (size - (end_index - start_index))
        test_output = [1] * (end_index - start_index) + [-1] * (end_index - start_index)
        return (training_input, training_output, test_input, test_output)

    # Convert the form of the dataset
    def transform_dataset(self, fold_index, fold_num, polarity_sents):
        (pos_sentence, neg_sentence) = self.extract_sentence(polarity_sents)
        (training_input, training_output, test_input, test_output) = self.divide_dataset(fold_index, fold_num, pos_sentence, neg_sentence)
        self.Y_train = training_output
        self.Y_test = test_output
        for sentence in training_input:
            instance = {}
            for word in sentence:
                self.word_features.add(word)
                instance[word] = 1
            self.X_train.append(instance)
        for sentence in test_input:
            instance = dict.fromkeys(sentence, 1)
            self.X_test.append(instance)

    # filtering the words in the dataset
    def filter_dataset(self, num):
        word_list = filter_words(self, num)
        self.word_features = set(word_list)
        for sentence in self.X_train:
            for word in sentence.keys():
                if word not in self.word_features:
                    del sentence[word]



"""baseline algorithm
"""
# build the dictionary of the positive words and negative words
path = "opinion-lexicon-English/"
pos_filename = "positive-words.txt"
neg_filename = "negative-words.txt"

filename = path + pos_filename
with open(filename, 'r') as f:
    lines = f.readlines()

pos_lexicons = set()
for line in lines:
    if line.startswith(";"):
        continue
    word = line.strip()
    pos_lexicons.add(word)

filename = path + neg_filename
with open(filename, 'r') as f:
    lines = f.readlines()

neg_lexicons = set()
for line in lines:
    if line.startswith(";"):
        continue
    word = line.strip()
    neg_lexicons.add(word)

# use the lexical ratio to determine whether it is negative or postive
def baseline_algorithm(pos_lexicons, neg_lexicons, test_input):
    pred_labels = []
    for i in range(len(test_input)):
        sentence = test_input[i].keys()
        pos_count = 0
        neg_count = 0
        for word in sentence:
            if word in pos_lexicons:
                pos_count += 1
            elif word in neg_lexicons:
                neg_count += 1
            else:
                continue
        if pos_count > neg_count:
            pred_labels.append(1)
        elif pos_count < neg_count:
            pred_labels.append(-1)
        else:
            pred_labels.append(0)
    return pred_labels

def calculate_confusionMatrix(pred_labels, labels):
    confusionMatrix = defaultdict(float)
    for i in range(len(labels)):
        if labels[i] == 1:
            if pred_labels[i] == 1:
                confusionMatrix['TP'] += 1.0
            elif pred_labels[i] == -1:
                confusionMatrix['FN'] += 1.0
            else:
                continue
        else:
            if pred_labels[i] == 1:
                confusionMatrix['FP'] += 1.0
            elif pred_labels[i] == -1:
                confusionMatrix['TN'] += 1.0
            else:
                continue
    return confusionMatrix

def evaluation(confusionMatrix):
    total = sum(confusionMatrix.values())
    TP = confusionMatrix['TP']
    TN = confusionMatrix['TN']
    FP = confusionMatrix['FP']
    FN = confusionMatrix['FN']
    accuracy = (TP + TN) / float(total)
    cc = TP * TN - FP * FN
    tmp = sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN))
    cc = float(cc) / float(tmp)
    return (accuracy, cc)









"""
Training the dataset
"""
# Multinomial naive bayes model
class Multinomial_NaiveBayesModel:

    def __init__(self):
        self.prior = defaultdict(float) #prior probaility p(c) = count(c) / total_count
        self.cond_prob = defaultdict(float) # conditional probability: p(word|c) = (count(c, word) + 1) / (count(c) + |v|)
        self.vocabulary = set()
        self.total_word_count = 0.0
        self.label_word_count = defaultdict(float) # count(c) key: label, value:  number of words
        self.tuple_word_count = defaultdict(float) # count(c, word) key: (label, word), value: number of words

    def train(self, dataset):
        self.vocabulary = dataset.word_features
        training_input = dataset.X_train
        training_output = dataset.Y_train
        # Count the frequency of the label and word
        for i in range(len(training_input)):
            label = training_output[i]
            sentence = training_input[i].keys()
            for word in sentence:
                if word not in self.vocabulary:
                    continue
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
        assert(self.total_word_count == (self.label_word_count[1] + self.label_word_count[-1])), "distribution of total is incorrect"
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
        for sentence in input:
            max_prob = 0.0
            pred_label = -1
            for label in [-1, 1]:
                pred_prob = self.prior[label]
                for word in sentence.keys():
                    item = (label, word)
                    if item not in self.cond_prob:
                        continue
                    tmp_prob = self.cond_prob[item]
                    pred_prob += tmp_prob
                pred_prob = exp(pred_prob)
                if pred_prob > max_prob:
                    max_prob = pred_prob
                    pred_label = label
            pred_output.append(pred_label)
        return pred_output



# Bernoulli Naive bayes model
class Bernoulli_NaiveBayesModel:
    vocabulary = set()  # set of words in the training set
    total_sent_count = 0.0  # count(sents)
    label_sent_count = defaultdict(float) # count(label) of sentences: count(label)
    tuple_sent_count = defaultdict(float) # count(label, word) of sentences: count(label, word)
    prior = defaultdict(float) # prior probability: p(c) = count(label) / count(sents)
    cond_prob = defaultdict(float) # conditional probability: p(word|c) = (count(label, word) + 1) / (count(label) + 2)

    def __init__(self):
        self.vocabulary = set()  # set of words in the training set
        self.total_sent_count = 0.0  # count(sents)
        self.label_sent_count = defaultdict(float) # count(label) of sentences: count(label)
        self.tuple_sent_count = defaultdict(float) # count(label, word) of sentences: count(label, word)
        self.prior = defaultdict(float) # prior probability: p(c) = count(label) / count(sents)
        self.cond_prob = defaultdict(float) # conditional probability: p(word|c) = (count(label, word) + 1) / (count(label) + 2)

    def train(self, dataset):
        self.vocabulary = dataset.word_features
        training_input = dataset.X_train
        training_output = dataset.Y_train
        for i in range(len(training_input)):
            sentence = training_input[i].keys()
            label = training_output[i]
            self.label_sent_count[label] += 1.0
            self.total_sent_count += 1.0
            sentence_set = set()
            for word in sentence:
                if word not in self.vocabulary:
                    continue
                item = (label, word)
                if item in sentence_set:
                    continue
                sentence_set.add(item)
                self.tuple_sent_count[item] += 1.0
        # Calculate the logaritmic value of probability
        label_set = set(training_output)
        for label in label_set:
            self.prior[label] = log(self.label_sent_count[label] / self.total_sent_count)
            for word in self.vocabulary:
                item = (label, word)
                self.cond_prob[item] = log((self.tuple_sent_count[item] + 1.0) / (self.label_sent_count[label] + 2.0))
        self.check_disribution()
        return

    def check_disribution(self):
        # Check the label
        assert(self.total_sent_count == (self.label_sent_count[1] + self.label_sent_count[-1])), "distribution of labels are incorrect"

    def classify(self, input):
        pred_output = []
        for sentence in input:
            max_prob = 0.0
            pred_label = -1
            for label in [-1, 1]:
                sentence_set = set()
                pred_prob = self.prior[label]
                for word in sentence.keys():
                    item = (label, word)
                    if item not in self.cond_prob:
                        continue
                    tmp_prob = self.cond_prob[item]
                    pred_prob += tmp_prob
                pred_prob = exp(pred_prob)
                if pred_prob > max_prob:
                    max_prob = pred_prob
                    pred_label = label
            pred_output.append(pred_label)
        return pred_output







"""Use the informtion gain to filter features
"""
# Calculate entropy
def CalculateEntropy(output_set):
    count = defaultdict(float)
    for label in output_set:
        count[label] += 1.0
    entropy = 0.0
    total = sum(count.values())
    for proportion in count.values():
        fraction = float(proportion) / float(total)
        entropy = entropy - fraction * np.log2(fraction)
    return entropy

# Calculate the information gain
def CalculateInfoGain(input_set, output_set):
    # The total entropy
    total_entropy = CalculateEntropy(output_set)
    # Divide the labels according to the value of the attribute in the input_set
    positive_output = []
    negative_output = []
    for i in range(len(input_set)):
        if input_set[i] == 1:
            positive_output.append(output_set[i])
        else:
            negative_output.append(output_set[i])
    # Calculate the reduction of the entropy, which is the information gain
    reduced_entropy = (float(len(positive_output)) / float(len(output_set))) * CalculateEntropy(positive_output)
    reduced_entropy += (float(len(negative_output)) / float(len(output_set))) * CalculateEntropy(negative_output)
    reduction_entropy = total_entropy - reduced_entropy
    return  reduction_entropy


# main function for filtering the features
def filter_words(dataset, num):
    feature_list = list(dataset.word_features)
    training_input = dataset.X_train
    training_output = dataset.Y_train
    info_gain_list = []
    for feature in feature_list:
        input_set = []
        output_set = list(training_output)
        for sentence in training_input:
            if feature not in sentence:
                input_set.append(0)
            else:
                input_set.append(1)
        info_gain = CalculateInfoGain(input_set, output_set)
        info_gain_list.append(info_gain)
    result_list = [feature for (info_gain, feature) in sorted(zip(info_gain_list, feature_list))]
    result_list.reverse()
    #reversed(result_list)
    print result_list[:10]
    return result_list[:num]













"""Use the max_ent model
"""
def classify_maxent(X_train, Y_train, X_test):
    training_input = X_train
    training_output = Y_train
    training_data = []
    for i in range(len(training_input)):
        training_data.append((training_input[i], training_output[i]))
    clf = MaxentClassifier.train(training_data)
    pred_labels = clf.classify_many(X_test)
    return pred_labels






# Main function for all the algorithm: run cross validation
def calculate_accuracy(pred_labels, labels):
    confusionMatrix = calculate_confusionMatrix(pred_labels, labels)
    (accuracy, cc) = evaluation(confusionMatrix)
    return accuracy



def calculate_test_accuracy(dataset):
    multinomial_naive_bayes_model = Multinomial_NaiveBayesModel()
    multinomial_naive_bayes_model.train(dataset)
    multinomial_accuracy = multinomial_naive_bayes_model.test(dataset.X_test, dataset.Y_test)

    bernoulli_naive_bayes_model = Bernoulli_NaiveBayesModel()
    bernoulli_naive_bayes_model.train(dataset)
    bernoulli_accuracy = bernoulli_naive_bayes_model.test(dataset.X_test, dataset.Y_test)

    return (multinomial_accuracy, bernoulli_accuracy)


def cross_validation(fold_num):
    polarityData = PolaritySents()
    dataset = nltk.corpus.product_reviews_2
    polarityData.preprocess_dataset(dataset)
    dataset = nltk.corpus.product_reviews_1
    polarityData.preprocess_dataset(dataset)
    multinomial_accuracy = []
    bernoulli_accuracy = []
    for i in range(fold_num):
        print i
        Preprocessed_dataset = Preprocess_Data()
        Preprocessed_dataset.transform_dataset(i, fold_num, polarityData)
        Preprocessed_dataset.filter_dataset(4000)

        test_accuracy = calculate_test_accuracy(Preprocessed_dataset)
        multinomial_accuracy.append(test_accuracy[0])
        bernoulli_accuracy.append(test_accuracy[1])

    print "accuracy of multinomial_accuracy: ", np.mean(multinomial_accuracy)
    print "accuracy of bernoulli_naive_bayes_model: ", np.mean(bernoulli_accuracy)



#cross_validation(10)







polarityData = PolaritySents()
dataset = nltk.corpus.product_reviews_2
polarityData.preprocess_dataset(dataset)
dataset = nltk.corpus.product_reviews_1
polarityData.preprocess_dataset(dataset)
Preprocessed_dataset = Preprocess_Data()
Preprocessed_dataset.transform_dataset(0, 10, polarityData)
Preprocessed_dataset.filter_dataset(4000)

"""baseline accuracy
"""
Y_pred = baseline_algorithm(pos_lexicons, neg_lexicons, Preprocessed_dataset.X_test)
accuracy = calculate_accuracy(Y_pred, Preprocessed_dataset.Y_test)
print "accuracy of baseline algorithm is: ", accuracy



multinomial_naive_bayes_model = Multinomial_NaiveBayesModel()
multinomial_naive_bayes_model.train(Preprocessed_dataset)
Y_pred = multinomial_naive_bayes_model.classify(Preprocessed_dataset.X_test)
multinomial_accuracy = calculate_accuracy(Y_pred, Preprocessed_dataset.Y_test)
print "accuracy of multinomial_accuracy: ", multinomial_accuracy

bernoulli_naive_bayes_model = Bernoulli_NaiveBayesModel()
bernoulli_naive_bayes_model.train(Preprocessed_dataset)
Y_pred = bernoulli_naive_bayes_model.classify(Preprocessed_dataset.X_test)
bernoulli_accuracy = calculate_accuracy(Y_pred, Preprocessed_dataset.Y_test)
print "accuracy of bernoulli_naive_bayes_model: ", bernoulli_accuracy

Y_pred = classify_maxent(Preprocessed_dataset.X_train, Preprocessed_dataset.Y_train, Preprocessed_dataset.X_test)
maxent_accuracy = calculate_accuracy(Y_pred, Preprocessed_dataset.Y_test)
print "accuracy of maxent model: ", maxent_accuracy
