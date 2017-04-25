# README

## Brief Introduction

In this project, a baseline algorithm, Multinomial Naive Bayes Model, Bernouli Naive Bayes Model, and the MaxEnt Model are implemented with different feature selection strategies. Through 10-fold cross validation and t-test between different aspects of the comparison, it is concluded that MaxEnt Model has the best performance with bigrams and unigrams feature together, but the POS tagging and the feature selection with information gain are unnecessary for the implementation.

## Dataset

### Dataset for dicionaries with different polarity in baseline algorithm

The corresponding dictionary can be found [here](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html) or you can just download the text file in the direct named 'opinion-lexicon-English'.

### Dataset for prediction

Since nltk module has two datasets named "product\_reviews\_1" and "product\_reviews\_2", these two datasets were selected as the training set and the test set for implementing the models. Among these two datasets, there are many product reviews for 14 different products: 

- digital camera: Canon G3
- digital camera: Nikon coolpix 4300
- celluar phone:  Nokia 6610
- mp3 player:     Creative Labs Nomad Jukebox Zen Xtra 40GB
- dvd player:     Apex AD2600 Progressive-scan DVD player
- ditigal camera: Canon PowerShot SD500
- digital camera: Canon S100
- baby bath:      Diaper Champ
- routers:        Hitachi router
- mp3 player:     ipod
- routers:        Linksys Router
- mp3 player:     MicroMP3
- cellphone:      Nokia 6600
- software:       norton

A example of the product reviews is below:

"camera[+3],size[+2]\#\#I'm in high school, and this camera is perfect for what I use it for, carrying it around in my pocket so I can take pictures whenever I want to, of my friends and of funny things that happen."

where the string after \#\# is the a sentence of the product reviews, the number in [ ] represents whether the sentence is positive or negative, and the word before [ ] represents the specific feature of the corresponding number in the [ ]. 

Since each sentence will have only one tag in this final project, the sentiment tag will be determine by calculating the sum of all the numbers in [] in a single sentence. If the result is larger than 0, then it is positive, but if the result is smaller than 0, then it is negative. We don't consider neutral altitude here.


## Techniques

### Models

- Baseline algorithm
- Naive Bayes Model

      - multinomial naive bayes model
      - bernoulli naive bayes model

- MaxEnt model

#### Baseline algorithm

The baseline algorithm used here is called lexical ratio algorithm. Two dictionaries named "negative-words" and "positive-words" were downloaded from website(website name: Opinion Mining, Sentiment Analysis, and Opinion Spam Detection) and stored into two text files. 

When predicating the sentiment label of a sentence, the number of the positive words and negative words were counted at first if the word was also in these two dictionaries. Suppose the count of these two types of words are: Count(pos) and Count(neg). 

If Count(pos) >= {Count(neg)}, the sentiment tag for this sentence is positive(the neural sentiment tag is not considered here). Otherwise, the sentiment tag for this sentence is negative.

#### Naive Bayes Model

**Multinomial Naive Byes Model**

In Multinomial Naive Bayes model, all the probability is calculated based on the count of the words in the dataset. The prior probability is below:

`P(c) = Count(words in class c)/ Count(words in dataset)`

and the conditional probability is below:

`P(t_k|c)=(Count(word t_k occurred in class c) + 1)/(Count(words in class c) + |V|)`

where `V` is the vocabulary of the training dataset.

Thus, the total probability of the sentence with tag c is:

`P(c|t_1...t_n) ~ P(c)P(t_1|c)P(t_2|c)...P(t_n|c)`

where `t_1`...`t_n` is a sentence with `n` words. 

Through this model, the sentiment tag of a sentence can be predicted.

**Bernoulli Naive Bayes Model**

In Bernoulli Naive Bayes Model, most of the probabilities were calculated based on the number of sentences in the dataset instead of the words. The prior probability is below:

`P(c) = Count(sentences in class c)/Count(sentences in dataset)`

and the conditional probability is below:

`P(t_k|c)=(Count(sentences with word t_k in class c) + 1)/(Count(words in class c) + 2)`

Through this model, the sentiment tag of a sentence can be predicted.

#### MaxEnt Model

MaxEnt Model can be thought as a kind of modification from naive bayes model. There are two significant differences between MaxEnt Model and the Naive Bayes Model:

1. MaxEnt Model considers features more widely, in other words, it can not only works with sequence based feature. If necessary, it can add more features based on the tags of the words like noun or verb, etc.
2. MaxEnt Model applys weight before all the features to indicate whether this feature is significant or not. Then, through some algorithm like gradient decent algorithm to optimize the value of the weight vector. Thus, the result of MaxEnt Model is usually more accurate comparing to the Naive Bayes Model.

Here, in order to use the MaxEnt Model, the MaxEnt classifier in nltk module was utilized with iteration of 50 times each time.

### Other teniques

- POS tagging: viterbi algorithm
- Feature Selection: information gain

Features can also affect the performance of the model significantly. According to the reference, most frequent feature is the unigram of the dataset. However, sometimes phrases are also important. Thus, bigram is important as well. Thus, we will implement the model with unigram alone and unigram plus bigram as the features.

After that, in order to add more features, POS tagging were applied with bigram hidden markov model to all the sentences in the dataset. After tagging all the words in the dataset, all the two-words phrase composed by an adjective word and a noun were extracted and selected as features added into the unigram models.

However, too many features will affect the efficiency of the algorithm especially for the MaxEnt Model. Thus, the information gain of all the features were calculated before feeding into the models. Since information gain can represent how useful this feature is for classification, all the features were ranked according to the value of the information gain. 2000 features with highest information gain were selected to feed into the models for classification.


## Evaluation

### t-test

The result of paired t-test is below(See Table~\ref{tab:models}):

| Model1 | Model2 | p-Value |
| ------ | :----: | ------: |
| maxent | bernoulli | <0.0001 |
| maxent | multinomial | <0.0001 |
| multinomial | bernoulli | 0.0478 |

### Overall results

The overall result for different models with or without POS tagging is below:

![totalreview](https://github.com/JoshuaW1990/SentimentAnalysis/blob/master/Images/totalreview.png?raw=true)




