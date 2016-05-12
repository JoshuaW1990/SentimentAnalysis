from nltk.classify import MaxentClassifier

train = [
    (dict(a=1,b=1,c=1), 'y'),
    (dict(a=1,b=1,c=1), 'x'),
    (dict(a=1,b=1), 'y'),
    (dict(b=1,c=1), 'x'),
    (dict(b=1,c=1), 'y'),
    (dict(c=1), 'y'),
    (dict(b=1), 'x'),
    (dict(), 'x'),
    (dict(b=1,c=1), 'y')
]

test = [
    (dict(a=1,c=1)), # unseen
    (dict(a=1)), # unseen
    (dict(b=1,c=1)), # seen 3 times, labels=y,y,x
    (dict(b=1, d=1)), # seen 1 time, label=x
]

clf = MaxentClassifier.train(train)

result = clf.classify_many(test)

print result
