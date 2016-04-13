#README

##Introduction
Implement the sentiment analysis of natural language process on the product review.

##Technique
###Model
Naive Bayes Model

###Feature
* Single word with high word score (chi-squared)
* Bigram with high word score

##Dataset
Import from nltk.corpus

##Bugs
- [X] Need to extract the sentence with positive or negative attitude from dataset.sents
      * We can use dataset.reviews() to extract each review and check each line of the reviews.

##TodoList
- [X] Import the Dataset
- [ ] Implement with single word as the Feature
- [ ] Implement with bigrams as the Feature
- [ ] Considering the tags of the word for the sentence
