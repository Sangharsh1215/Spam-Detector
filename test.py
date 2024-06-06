
import pickle
import string
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')


def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

with open('sms_spam_pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)


mesg = ["hi are you free today?"]

prediction = pipeline.predict(mesg)


print("Spam" if prediction[0] == "spam" else "Ham")
