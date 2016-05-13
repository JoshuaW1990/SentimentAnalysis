import nltk
import nltk.corpus
from nltk.classify import MaxentClassifier
import numpy as np
from random import shuffle
from collections import defaultdict
from math import log, exp, sqrt

from nltk.corpus import treebank
from nltk.tag.util import untag  # Untags a tagged sentence.



"""POS tagging for the dataset
"""
unknown_token = "<UNK>"  # unknown word token.
start_token = "<S>"  # sentence boundary token.
end_token = "</S>"  # sentence boundary token.

#Remove trace tokens and tags from the treebank as these are not necessary.
def TreebankNoTraces():
    return [[x for x in sent if x[1] != "-NONE-"] for sent in treebank.tagged_sents()]

#the function for preprocess the text
def PreprocessText(dataset, vocab):
    new_set = []
    for sentence in dataset:
        if len(sentence) == 0:
            print sentence, dataset.index(sentence)
        tmpSentence = list(sentence)
        for i in range(len(tmpSentence)):
            if tmpSentence[i][0] not in vocab:
                tmpSentence[i] = (unknown_token, sentence[i][1])
        tmp_sent = [(start_token, start_token)] + tmpSentence + [(end_token, end_token)]
        new_set.append(tmp_sent)
    return new_set

# get the vocabulary
def PreprocessVocab(dataset):
    vocab_dict = defaultdict(int)
    vocabulary = set([])
    for sentence in dataset:
        for word in sentence:
            vocab_dict[word[0]] += 1
    for word in vocab_dict.iterkeys():
        if vocab_dict[word] > 1:
            vocabulary.add(word)
    return vocabulary

class BigramHMM:
    def __init__(self):
        """ Implement:
        self.transitions, the A matrix of the HMM: a_{ij} = P(t_j | t_i)
        self.emissions, the B matrix of the HMM: b_{ii} = P(w_i | t_i)
        self.dictionary, a dictionary that maps a word to the set of possible tags
        """
        self.transitions = defaultdict(float)
        self.emissions = defaultdict(float)
        self.dictionary = defaultdict(set)

    def Train(self, training_set):
        """
        1. Estimate the A matrix a_{ij} = P(t_j | t_i)
        2. Estimate the B matrix b_{ii} = P(w_i | t_i)
        3. Compute the tag dictionary
        """
        unigram_tag = defaultdict(float)
        for sentence in training_set:
            for i in range(len(sentence)):
                word = sentence[i][0]
                tag = sentence[i][1]
                if word not in self.dictionary:
                    self.dictionary[word] = set()
                self.dictionary[word].add(tag)
                unigram_tag[tag] += 1
                self.emissions[sentence[i]] += 1
                if i < len(sentence) - 1:
                    self.transitions[(tag, sentence[i + 1][1])] += 1
        for bigram in self.transitions.keys():
            self.transitions[bigram] = self.transitions[bigram] / unigram_tag[bigram[0]]
        for unigram in self.emissions.keys():
            self.emissions[unigram] = self.emissions[unigram] / unigram_tag[unigram[1]]
        return None


    def ComputePercentAmbiguous(self, data_set):
        """ Compute the percentage of tokens in data_set that have more than one tag according to self.dictionary. """
        tag_dict = defaultdict(set)
        total_token = 0.0
        ambiguous_token = 0.0
        for sentence in data_set:
            for word in sentence:
                total_token += 1.0
                if len(self.dictionary[word[0]]) > 1:
                    ambiguous_token += 1.0
        percent_ambiguous = ambiguous_token / total_token
        print "There are %s tags for unknown_token." %len(self.dictionary[unknown_token])
        print "The tags for the unknown token are ", list(self.dictionary[unknown_token])
        return (100 * percent_ambiguous)

    def JointProbability(self, sent):
        """ Compute the joint probability of the words and tags of a tagged sentence. """
        probability = 1
        for i in range(1, len(sent)):
            current_tag = sent[i][1]
            prev_tag = sent[i-1][1]
            probability = probability * self.transitions[(prev_tag, current_tag)] * self.emissions[sent[i]]
        return probability

    def findMax(self, viterbi_dict, current_state, current_word, state):
        maxViterbi = 0.0
        maxPrevState = state[0]
        for prev_state in state:
            tmp_viterbi = viterbi_dict[prev_state] * self.transitions[(prev_state, current_state)] * self.emissions[(current_word, current_state)]
            if maxViterbi < tmp_viterbi:
                maxViterbi = tmp_viterbi
                maxPrevState = prev_state
        return (maxViterbi, maxPrevState)

    def Viterbi(self, sent):
        """ Find the probability and identity of the most likely tag sequence given the sentence. """
        # Preprocess to get the list of states
        tmp_sent = sent[1:-1]
        state = set()
        for word in tmp_sent:
            state.update(self.dictionary[word[0]])
        state = list(state)
        viterbi_matrix = []
        backpointers = []
        #Initialization step
        tag_dict = defaultdict(float)
        back_dict = defaultdict(str)
        for current_state in state:
            prev_state = sent[0][1]
            current_word = sent[1][0]
            tag_dict[current_state] = self.transitions[(prev_state, current_state)] * self.emissions[(current_word, current_state)]
            back_dict[current_state] = prev_state
        viterbi_matrix.append(tag_dict)
        backpointers.append(back_dict)
        #Recursion step
        for t in range(1, len(tmp_sent)):
            tag_dict = defaultdict(float)
            back_dict = defaultdict(str)
            current_word = tmp_sent[t][0]
            for current_state in state:
                (tag_dict[current_state], back_dict[current_state]) = self.findMax(viterbi_matrix[-1], current_state, current_word, state)
            viterbi_matrix.append(tag_dict)
            backpointers.append(back_dict)
        #termination step
        (current_word, current_state) = sent[-1]
        tag_dict = defaultdict(float)
        back_dict = defaultdict(str)
        (tag_dict[current_state], back_dict[current_state]) = self.findMax(viterbi_matrix[-1], current_state, current_word, state)
        viterbi_matrix.append(tag_dict)
        backpointers.append(back_dict)
        #Get the backtrace by the backpointers
        backpointers.reverse()
        backtrace = [sent[-1][1]]
        for search_dict in backpointers:
            tag = search_dict[backtrace[-1]]
            backtrace.append(tag)
        backtrace.reverse()
        if viterbi_matrix[-1].values()[0] == 0:
            return "incorrect"
        else:
            return backtrace



    def Predict(self, test_set):
        """ Use Viterbi and predict the most likely tag sequence for every sentence. Return a re-tagged test_set. """
        predict_set = []
        flag = 1
        for sentence in test_set:
            backtrace = self.Viterbi(sentence)
            if backtrace == "incorrect":
                predict_set.append(sentence)
                continue
            predict_sentence = list(sentence)
            for i in range(len(predict_sentence)):
                predict_sentence[i] = (predict_sentence[i][0], backtrace[i])
            predict_set.append(predict_sentence)
        return predict_set


    def ConfusionMatrix(self, test_set, test_set_predicted):
        #preprocess the data
        tag_set = set()
        for tags in self.dictionary.values():
            tag_set.update(tags)
        tag_list = list(tag_set)
        size = len(tag_list)
        confusion_matrix = [[0.0 for x in range(size)] for y in range(size)]  # In the confusion matrix, the first index is real tag, second is the predict tag
        total_tagerror = 0.0
        # Updating the confusion matrix
        for i in range(len(test_set)):
            real_sent = test_set[i]
            predict_sent = test_set_predicted[i]
            if test_set_predicted[i] == "incorrect":
                continue
            for j in range(len(real_sent)):
                real_tag = real_sent[j][1]
                predict_tag = predict_sent[j][1]
                if real_tag != predict_tag:
                    total_tagerror += 1.0
                    real_position = tag_list.index(real_tag)
                    predict_position = tag_list.index(predict_tag)
                    confusion_matrix[real_position][predict_position] += 1.0
        for i in range(size):
            for j in range(size):
                confusion_matrix[i][j] = 100 * confusion_matrix[i][j] / total_tagerror
        # Extract the most confused classes
        error_list = []
        for num_list in confusion_matrix:
            for num in num_list:
                if num > 5:
                    error_list.append(num)
        # print the output
        error_list.sort()
        error_list.reverse()
        for num in error_list:
            print "---------------------------------"
            print "The error percentage is %.2f%%." %num,
            for i in range(size):
                if num in confusion_matrix[i]:
                    real_tag = tag_list[i]
                    j = confusion_matrix[i].index(num)
                    predict_tag = tag_list[j]
                    break
            print "The tags are ", real_tag, "->", predict_tag
        return None

def ComputeAccuracy(test_set, test_set_predicted):
    """ Using the gold standard tags in test_set, compute the sentence and tagging accuracy of test_set_predicted. """
    correct_sent = 0
    correct_tag = 0
    total_sent = len(test_set)
    total_tag = 0
    for i in range(len(test_set)):
        if test_set_predicted[i] == "incorrect":
            continue
        total_tag += len(test_set[i])
        total_tag -= 2
        flag = 1
        for j in range(1, (len(test_set[i]) -1)):
            if test_set[i][j][1] == test_set_predicted[i][j][1]:
                correct_tag += 1
            else:
                flag = 0
        if flag == 1:
            correct_sent += 1
    sent_accuracy = float(correct_sent) / float(total_sent)
    tag_accuracy = float(correct_tag) / float(total_tag)
    print "sentence accuracy is %.2f%%." %(100 * sent_accuracy)
    print "tag accuracy is %.2f%%." %(100 * tag_accuracy)
    return None




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
            if sents[i] == []:
                continue
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
class Preprocess_Data_unigram:
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
        #shuffle(pos_sentence)
        neg_sentence = polarity_sents.negSents
        #shuffle(neg_sentence)

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


class Preprocess_Data_unigram_POS:
    def __init__(self):
        self.X_train =[]
        self.Y_train = []
        self.X_test = []
        self.Y_test = []
        self.word_features = set()

    def extract_sentence(self, polarity_sents, vocab):
        # Get the minimum value of the size
        pos_size = len(polarity_sents.posSents)
        neg_size = len(polarity_sents.negSents)
        min_size = min(pos_size, neg_size)
        # shuffle the dataset
        pos_sentence = polarity_sents.posSents
        #shuffle(pos_sentence)
        neg_sentence = polarity_sents.negSents
        #shuffle(neg_sentence)

        pos_sentence = pos_sentence[:min_size]
        neg_sentence = neg_sentence[:min_size]

        new_pos_sentence = []
        new_neg_sentence = []
        for i in range(len(pos_sentence)):
            sentence1 = pos_sentence[i]
            tmp_list1 = [1 for j in range(len(sentence1))]
            tmp_sentence1 = zip(sentence1, tmp_list1)
            new_pos_sentence.append(tmp_sentence1)
            sentence2 = neg_sentence[i]
            tmp_list2 = [1 for j in range(len(sentence2))]
            tmp_sentence2 = zip(sentence2, tmp_list2)
            new_neg_sentence.append(tmp_sentence2)

        pos_sentence_prep = PreprocessText(new_pos_sentence, vocab)
        neg_sentence_prep = PreprocessText(new_neg_sentence, vocab)
        return (pos_sentence_prep, neg_sentence_prep)

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
    def transform_dataset(self, fold_index, fold_num, polarity_sents, bigram_hmm, vocab):
        (pos_sentence, neg_sentence) = self.extract_sentence(polarity_sents, vocab)
        (training_input, training_output, test_input, test_output) = self.divide_dataset(fold_index, fold_num, pos_sentence, neg_sentence)
        self.Y_train = training_output
        self.Y_test = test_output
        adjective_words = set(['JJ', 'JJR', 'JJS'])
        noun_words = set(['NN', 'NNP', 'NNPS', 'NNS'])
        pred_training_sents = bigram_hmm.Predict(training_input)
        pred_test_sents = bigram_hmm.Predict(test_input)
        for sentence in pred_training_sents:
            instance = {}
            for i in range(1, len(sentence)):
                #print pred_tags[i], i
                #print pred_tags
                current_word = sentence[i][0]
                current_tag = sentence[i][1]
                prev_word = sentence[i - 1][0]
                prev_tag = sentence[i - 1][1]
                self.word_features.add(current_word)
                instance[current_word] = 1
                if prev_tag in adjective_words and current_tag in noun_words:
                    bigram = (prev_word, current_word)
                    instance[bigram] = 1
            self.X_train.append(instance)
        for sentence in pred_test_sents:
            instance = {}
            for i in range(1, len(sentence)):
                #print pred_tags[i], i
                #print pred_tags
                current_word = sentence[i][0]
                current_tag = sentence[i][1]
                prev_word = sentence[i - 1][0]
                prev_tag = sentence[i - 1][1]
                self.word_features.add(current_word)
                instance[current_word] = 1
                if prev_tag in adjective_words and current_tag in noun_words:
                    bigram = (prev_word, current_word)
                    instance[bigram] = 1
            self.X_test.append(instance)

    # filtering the words in the dataset
    def filter_dataset(self, num):
        word_list = filter_words(self, num)
        self.word_features = set(word_list)
        for sentence in self.X_train:
            for word in sentence.keys():
                if word not in self.word_features:
                    del sentence[word]



class Preprocess_Data_bigram:
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
        #shuffle(pos_sentence)
        neg_sentence = polarity_sents.negSents
        #shuffle(neg_sentence)

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
            for i in range(len(sentence)):
                word = sentence[i]
                self.word_features.add(word)
                instance[word] = 1
                if i == 0:
                    continue
                bigram = (sentence[i - 1], word)
                self.word_features.add(bigram)
                instance[bigram] = 1
            self.X_train.append(instance)
        for sentence in test_input:
            instance = {}
            for i in range(len(sentence)):
                word = sentence[i]
                instance[word] = 1
                if i == 0:
                    continue
                bigram = (sentence[i - 1], word)
                instance[bigram] = 1
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
        Preprocessed_dataset = Preprocess_Data_bigram()
        Preprocessed_dataset.transform_dataset(i, fold_num, polarityData)
        Preprocessed_dataset.filter_dataset(4000)

        test_accuracy = calculate_test_accuracy(Preprocessed_dataset)
        multinomial_accuracy.append(test_accuracy[0])
        bernoulli_accuracy.append(test_accuracy[1])

    print "accuracy of multinomial_accuracy: ", np.mean(multinomial_accuracy)
    print "accuracy of bernoulli_naive_bayes_model: ", np.mean(bernoulli_accuracy)



#cross_validation(10)



treebank_tagged_sents = TreebankNoTraces()  # Remove trace tokens.
training_set = treebank_tagged_sents[:3000]  # This is the train-test split that we will use.
test_set = treebank_tagged_sents[3000:]

# Transform the data sets by eliminating unknown words and adding sentence boundary tokens.
vocabulary = PreprocessVocab(training_set)
training_set_prep = PreprocessText(training_set, vocabulary)
test_set_prep = PreprocessText(test_set, vocabulary)

bigram_hmm = BigramHMM()
bigram_hmm.Train(training_set_prep)

# POS tagging for the test dataset
test_set_predicted_bigram_hmm = bigram_hmm.Predict(test_set_prep)
print "--- Bigram HMM accuracy ---"
ComputeAccuracy(test_set_prep, test_set_predicted_bigram_hmm)



polarityData = PolaritySents()
dataset = nltk.corpus.product_reviews_2
polarityData.preprocess_dataset(dataset)
dataset = nltk.corpus.product_reviews_1
polarityData.preprocess_dataset(dataset)
Preprocessed_dataset = Preprocess_Data_unigram()
Preprocessed_dataset.transform_dataset(0, 10, polarityData)
#Preprocessed_dataset.transform_dataset(0, 10, polarityData, bigram_hmm, vocabulary
#Preprocessed_dataset.filter_dataset(4000)

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

"""
Y_pred = classify_maxent(Preprocessed_dataset.X_train, Preprocessed_dataset.Y_train, Preprocessed_dataset.X_test)
maxent_accuracy = calculate_accuracy(Y_pred, Preprocessed_dataset.Y_test)
print "accuracy of maxent model: ", maxent_accuracy
"""