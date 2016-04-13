import nltk
import nltk.corpus

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
