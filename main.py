
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pickle


messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]
messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', names=['label', 'message'])


def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.3)


pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])


pipeline.fit(msg_train, label_train)


with open('sms_spam_pipeline.pkl', 'wb') as file:
    pickle.dump(pipeline, file)


predictions = pipeline.predict(msg_test)
from sklearn.metrics import classification_report
print(classification_report(label_test, predictions))
