"""
Import Dataset
"""
import nltk
import nltk.corpus


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

# Implementing the naive bayes model with only single word
def Classifier_SingleWord:
    def __init__(self):
        




polarityData = PolaritySents()
dataset = nltk.corpus.product_reviews_2.raw()
polarityData.preprocess_dataset(dataset)
dataset = nltk.corpus.product_reviews_1.raw()
polarityData.preprocess_dataset(dataset)
