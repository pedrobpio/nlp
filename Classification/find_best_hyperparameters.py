import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


class Bag_of_words():
    def init(self):
        self.vectorized_texts = None
    
    # method that counts the words and transforms it to vectors
    def preprocessing(self, text_array):
        #cout the the classes
        vectorizer = CountVectorizer()
        self.vectorized_texts = vectorizer.fit_transform(text_array)

        return self.vectorized_texts

    def train_best(self, text_array, labels, NB):
        self.preprocessing(text_array)
        X_train, X_test, y_train, y_test = train_test_split(self.vectorized_texts, labels, test_size=0.2, random_state=42)
        
        NB.fit(X_train, y_train)
        results = NB.predict(X_test)
        report = classification_report(y_test, results)
        micro_f1 = f1_score(y_test, results, average='micro')
        return report, micro_f1

    def grid_search(self, text_array, labels,  params = {'alpha': [0.1, 0.4, 0.7, 1, 1.3, 1.5]}):
        self.preprocessing(text_array)
        X_train, X_test, y_train, y_test = train_test_split(self.vectorized_texts, labels, test_size=0.2, random_state=42)

        NB = MultinomialNB()
       
        grid = GridSearchCV(NB, params, n_jobs=-1, cv =3)
        grid.fit(X_train, y_train)
        return grid

    

class Logistic_regression_classifier():
    def __init__(self):
        self.Tfid_values = None
    
    def preprocessing(self, text_array):
        # vectorize the values and extract features based on tfid
        vectorizer = TfidfVectorizer()
        self.vectorized_texts = vectorizer.fit_transform(text_array)

        return self.vectorized_texts
    
    def train_best(self, text_array, labels, LR):
        self.preprocessing(text_array)
        X_train, X_test, y_train, y_test = train_test_split(self.vectorized_texts, labels, test_size=0.2, random_state=42)
        
        LR.fit(X_train, y_train)
        results = LR.predict(X_test)
        report = classification_report(y_test, results)
        micro_f1 = f1_score(y_test, results, average='micro')
        return report, micro_f1

    def grid_search(self, text_array, labels, params = { 'C': [1, 5, 10], 'max_iter': [10, 50, 100] }):
        self.preprocessing(text_array)
        X_train, X_test, y_train, y_test = train_test_split(self.vectorized_texts, labels, test_size=0.2, random_state=42)

        LR = LR = LogisticRegression()
        grid = GridSearchCV(LR, params, n_jobs=-1, cv =2)
        grid.fit(X_train, y_train)
        return grid


class KNN_classifier():
    def __init__(self):
        self.Tfid_values = None
    
    def preprocessing(self, text_array):
        # vectorize the values and extract features based on tfid
        vectorizer = TfidfVectorizer()
        self.vectorized_texts = vectorizer.fit_transform(text_array)

        return self.vectorized_texts
    
    def train_best(self, text_array, labels, KNN):
        self.preprocessing(text_array)
        X_train, X_test, y_train, y_test = train_test_split(self.vectorized_texts, labels, test_size=0.2, random_state=42)
        
        KNN.fit(X_train, y_train)
        results = KNN.predict(X_test)
        report = classification_report(y_test, results)
        micro_f1 = f1_score(y_test, results, average='micro')
        return report, micro_f1

    def grid_search(self, text_array, labels, params = { 'n_neighbors': [4, 8, 12, 16, 20, 24, 28, 32]}):
        self.preprocessing(text_array)
        X_train, X_test, y_train, y_test = train_test_split(self.vectorized_texts, labels, test_size=0.2, random_state=42)

        KNN = KNeighborsClassifier()
        grid = GridSearchCV(KNN, params, n_jobs=-1, cv =2)
        grid.fit(X_train, y_train)
        return grid


        