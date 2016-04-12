
import nltk
import nltk.corpus
from random import shuffle
from collections import defaultdict

"""
Import Dataset
"""
class PolaritySents:
    def __init__(self):
        self.posSents = []
        self.negSents = []

    #example: dataset = nltk.corpus.product_reviews_2.raw()
    def preprocess_dataset(self, dataset):
        lines = dataset.encode('utf8').split('\n')
        count = 1
        for line in lines:
            count += 1
            if line.startswith('[t]') or line.startswith('*'):
                continue
            string = line.split("##")
            if string[0] == '' or len(string) < 2:
                continue
            attitude = string[0].split('[')
            sent_attitude = 0
            for item in attitude:
                if item.endswith(']') and item[-2] < '9' and item[-2] > '0':
                    try:
                        num = int(item[:-1])
                        sent_attitude += num
                    except:
                        continue
                else:
                    continue
            if sent_attitude > 0:
                self.posSents.append(string[1])
            elif sent_attitude < 0:
                self.negSents.append(string[1])
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
        training_input = []
        training_output = []
        test_input = []
        test_output = []

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
        if fold_index > fold_num:
            print "error when dividing the dataset in cross validation"
            return None
        size = len(pos_sentence)
        fold_size = int(size / fold_num)
        return
        



polarityData = PolaritySents()
dataset = nltk.corpus.product_reviews_2.raw()
polarityData.preprocess_dataset(dataset)
dataset = nltk.corpus.product_reviews_1.raw()
polarityData.preprocess_dataset(dataset)
