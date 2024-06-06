import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]

import pandas as pd 
messages=pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', names=['label','message'])

import string
def text_process(mess):
    nopunc=[char for char in mess if char not in string.punctuation]
    nopunc=''.join(nopunc)
    return[word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

from sklearn.feature_extraction.text import CountVectorizer
bow_transformer=CountVectorizer(analyzer=text_process).fit(messages['message'])
messages_bow=bow_transformer.transform(messages['message'])

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer=TfidfTransformer().fit(messages_bow)

messages_tfidf=tfidf_transformer.transform(messages_bow)
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf,messages['label'])

from sklearn.model_selection import train_test_split
msg_train,msg_test,label_train,label_test=train_test_split(messages['message'], messages['label'],test_size=0.3)

from sklearn.pipeline import Pipeline
pipeline=Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier', MultinomialNB())

])

pipeline.fit(msg_train,label_train)


predictions=pipeline.predict(msg_test)

from sklearn.metrics import classification_report
print(classification_report(label_test,predictions))
print("Hello")