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

class Param_optmizer():

    def __init__(self) -> None:
        """
        Initializes the class with empty attributes to store data and models.
        """
        self.train_data = None
        self.train_label = None
        self.test_data = None
        self.test_label = None
        self.tfidf_train = None
        self.tfidf_test = None
        self.bag_of_words_train = None
        self.bag_of_words_test = None
        self.NB = None
        self.KNN = None
        self.LR = None
        self.NB_grid = None
        self.LR_grid = None
        self.KNN_grid = None
    
    def get_NB_grid(self):
        """
        Returns the stored GridSearchCV object for Naive Bayes.
        """
        return self.NB_grid

    def get_LR_grid(self):
        """
        Returns the stored GridSearchCV object for Logistic Regression.
        """
        return self.LR_grid

    def get_KNN_grid(self):
        """
        Returns the stored GridSearchCV object for K-Nearest Neighbors.
        """
        return self.KNN_grid

    
    def train_test_split(self, data, labels, test_size=0.2, random_state=42):
        """
        Splits the data and labels into training and testing sets using scikit-learn's train_test_split function.
        Prints informative messages about the split.

        Args:
            data: The data to be split.
            labels: The corresponding labels for the data.
            test_size: The proportion of data to be used for testing (default: 0.2).
            random_state: The random seed for splitting (default: 42).
        """
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
        self.train_data = X_train
        self.test_data = X_test
        self.train_label = y_train
        self.test_label = y_test

        print(f'The dataset was splited into train ({(1-test_size)*100}%) and test ({(test_size)*100}%)')
        print(f'dataset: {data.shape[0]} rows')
        print(f'train set: {self.train_data.shape[0]} rows')
        print(f'test set: {self.test_data.shape[0]} rows')

    
    def preprocessing(self):
        """
        Performs preprocessing on the training and testing data if they exist.
        Creates Bag-of-Words and TF-IDF representations.
        """
        if (self.train_data is not None):
            print(f'train set: {self.train_data.shape[0]} rows')
            print(f'test set: {self.test_data.shape[0]} rows')
            print("Train and test data found, creating bag of words")
            vectorizer = CountVectorizer()
            self.bag_of_words_train = vectorizer.fit_transform(self.train_data)
            self.bag_of_words_test = vectorizer.transform(self.test_data)
            print("the bag of words was created")

            print("creatting TfiDF")

            vectorizer = TfidfVectorizer()
            self.tfidf_train = vectorizer.fit_transform(self.train_data)
            self.tfidf_test = vectorizer.transform(self.test_data)
            print("the TfiDF vector was created.")

        else:
            print("Please, add the train and test data first")

    def NB_Grid_search(self, params = {'alpha': [0.1, 0.4, 0.7, 1, 1.3, 1.5]} ):
        """
        Performs Grid Search on Naive Bayes to find the best hyperparameters.

        Grid Search exhaustively tries all combinations of the provided parameters
        using cross-validation to evaluate each combination's performance. The best
        performing combination is then selected.

        Args:
            params (dict): A dictionary of parameters to search over. The key is the
                           parameter name (e.g., 'alpha'), and the value is a list of
                           values to try. Defaults to a predefined range of alpha values.

        Returns:
            GridSearchCV: The fitted GridSearchCV object containing the results of the search.
                         Returns None if bag of words was not computed.
        """
        if self.bag_of_words_train is not None:
            print("All set, starting to get the best params")
            NB = MultinomialNB()
            grid = GridSearchCV(NB, params, n_jobs=-1, cv =3)
            grid.fit(self.bag_of_words_train, self.train_label)
            print(grid)
            self.NB_grid = grid
            self.NB = grid.best_estimator_
            return grid
        else:
            print("Please, build the bag of words with the preprocessing mehtod before doing the grid search")
        return None

    def LR_Grid_search(self, params = { 'C': [1, 5, 10], 'max_iter': [5, 20, 50] } ):
        """
        Performs Grid Search on Logistic Regression to find the best hyperparameters.

        Similar to NB_Grid_search, this method exhaustively searches over the provided
        parameter grid using cross-validation.

        Args:
            params (dict): Dictionary of parameters to search.

        Returns:
            GridSearchCV: The fitted GridSearchCV object. Returns None if tfidf was not computed.
        """
        if self.tfidf_train is not None:
            print("All set, starting to get the best params")
            LR = LogisticRegression()
            grid = GridSearchCV(LR, params, n_jobs=-1, cv =3)
            grid.fit(self.tfidf_train, self.train_label)
            print(grid)
            self.LR_grid = grid
            self.LR = grid.best_estimator_
            return grid
        else:
            print("Please, build the bag of words with the preprocessing mehtod before doing the grid search")

    def KNN_Grid_search(self, params = { 'n_neighbors': [4, 8, 12, 16, 20, 24, 28, 32]}):
        """
        Performs Grid Search on K-Nearest Neighbors to find the best hyperparameter.

        Exhaustively searches the parameter space using cross-validation.

        Args:
            params (dict): Dictionary of parameters to search.

        Returns:
            GridSearchCV: The fitted GridSearchCV object. Returns None if tfidf was not computed.
        """
        if self.tfidf_train is not None:
            print("All set, starting to get the best params")
            KNN = KNeighborsClassifier()
            grid = GridSearchCV(KNN, params, n_jobs=-1, cv =3)
            grid.fit(self.tfidf_train, self.train_label)
            print(grid)
            self.KNN_grid = grid
            self.KNN = grid.best_estimator_
            return grid

        else:
            print("Please, build the bag of words with the preprocessing mehtod before doing the grid search")

    def train_and_run_best_NB(self, NB = None):
        """
        Trains the best Naive Bayes model (found by Grid Search) and evaluates it.

        If a NB model is provided, it will be trained and used. Otherwise, the
        model stored in self.NB (the result of Grid Search) will be used.

        Args:
            NB: A Naive Bayes model instance. If None, uses self.NB.

        Returns:
            tuple: A tuple containing the classification report and micro-averaged F1 score.
                   Returns None if NB model is not available.
        """
        if NB is None:
            NB = self.NB
        if NB is not None:
            NB.fit(self.bag_of_words_train, self.train_label)
            results = NB.predict(self.bag_of_words_test)
            report = classification_report(self.test_label, results)
            micro_f1 = f1_score(self.test_label, results, average='micro')
            return report, micro_f1
        return None

    def train_and_run_best_LR(self, LR = None):
        """
        Trains the best Logistic Regression model (found by Grid Search) and evaluates it.

        If a LR model is provided, it will be trained and used. Otherwise, the
        model stored in self.LR (the result of Grid Search) will be used.

        Args:
            LR: A Logistic Regression model instance. If None, uses self.LR.

        Returns:
            tuple: A tuple containing the classification report and micro-averaged F1 score.
                   Returns None if LR model is not available.
        """
        if LR is None:
            LR = self.LR
        if LR is not None:
            LR.fit(self.tfidf_train, self.train_label)
            results = LR.predict(self.tfidf_test)
            report = classification_report(self.test_label, results)
            micro_f1 = f1_score(self.test_label, results, average='micro')
            return report, micro_f1
        return None
    
    def train_and_run_best_KNN(self, KNN = None):
        """
        Trains the best KNN model (found by Grid Search) and evaluates it.

        If a KNN model is provided, it will be trained and used. Otherwise, the
        model stored in self.KNN (the result of Grid Search) will be used.

        Args:
            KNN: A KNN model instance. If None, uses self.KNN.

        Returns:
            tuple: A tuple containing the classification report and micro-averaged F1 score.
                   Returns None if KNN model is not available.
        """
        if KNN is None:
            KNN = self.KNN
        if KNN is not None:
            KNN.fit(self.tfidf_train, self.train_label)
            results = KNN.predict(self.tfidf_test)
            report = classification_report(self.test_label, results)
            micro_f1 = f1_score(self.test_label, results, average='micro')
            return report, micro_f1
        return None
