# fake news prediction 
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

import nltk
nltk.download("stopwords")
print(stopwords.words("english"))

dataset = pd.read_csv("./train.csv")

dataset.head()

dataset.isnull().sum()

dataset = dataset.fillna(" ")

dataset.isnull().sum()

dataset["content"] = dataset["author"] +" "+ dataset["title"]


print(dataset["content"])



X = dataset.drop(columns = "label",axis=1)
Y = dataset["label"]

port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

dataset['content'] = dataset['content'].apply(stemming)
print(dataset['content'])

X = dataset['content'].values
print(X)

Y = dataset['label'].values
print(Y)

vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)
print(X)

X_train , X_test , Y_train , Y_test =  train_test_split(X,Y,random_state = 2 ,test_size = 0.2,stratify=Y)
print(X.shape , X_test.shape , X_train.shape)
print(Y.shape , Y_test.shape , Y_train.shape)


model = LogisticRegression()
model.fit(X_train , Y_train)


# mdoel evaluation

train_data_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(train_data_prediction , Y_train)
print(training_data_accuracy*100)


test_data_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(test_data_prediction , Y_test)
print(test_data_accuracy*100)


with open("Logestic_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved to disk.")

# Save the vectorizer to disk
with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)
