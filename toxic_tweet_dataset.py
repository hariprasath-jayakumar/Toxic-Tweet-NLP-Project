**Importing Required Lib**

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

import spacy
import re

"""**Loading DataSet**"""

df = pd.read_csv("/content/FinalBalancedDataset.csv", encoding="utf-8")
df.drop(df.columns[0], axis=1, inplace=True)

"""**Checking For Null Data**"""

df.isnull().sum()

"""**Load english language model and create nlp object from it**"""

nlp = spacy.load("en_core_web_sm")

"""**Defining a function to preprocssing the Text Data**"""

def preprocess(tweet):
    doc = nlp(tweet)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct or not token.text.isalnum():
            continue
        filtered_tokens.append(token.lemma_.lower())

    return " ".join(filtered_tokens)

"""**Preprocessing our data**"""

df['preprocessed'] = df['tweet'].apply(preprocess)

df.isnull().sum()

"""**Fit & Transforming with Classifiers**"""

#Bag_of_Words
vectorizer_bow = CountVectorizer()
bow_features = vectorizer_bow.fit_transform(df['preprocessed'])

#TF_IDF
vectorizer_tfidf = TfidfVectorizer()
tfidf_features = vectorizer_tfidf.fit_transform(df['preprocessed'])

"""**Train & Test Split**"""

x_train_bow, x_test_bow, y_train, y_test = train_test_split(bow_features, df['Toxicity'], test_size=0.2, random_state=42)

x_train_tfidf, x_test_tfidf,y_train,y_test = train_test_split(tfidf_features, df['Toxicity'], test_size=0.2, random_state=42)

"""**Classifier Fitting & Object Creation**

> **Bag_of_Words**
"""

# Decision Trees
dt_classifier_bow = DecisionTreeClassifier(random_state=42)
dt_classifier_bow.fit(x_train_bow, y_train)

# Random Forest
rf_classifier_bow = RandomForestClassifier(random_state=42)
rf_classifier_bow.fit(x_train_bow, y_train)

# Naive Bayes
nb_classifier_bow = MultinomialNB()
nb_classifier_bow.fit(x_train_bow, y_train)

# K-NN Classifier
knn_classifier_bow = KNeighborsClassifier(n_neighbors=5)
knn_classifier_bow.fit(x_train_bow, y_train)

# SVM Classifier
svm_classifier_bow = SVC(probability=True, random_state=42)
svm_classifier_bow.fit(x_train_bow, y_train)

"""

> **BOW Prediction**


"""

y_pred_dt_bow = dt_classifier_bow.predict(x_test_bow)
y_pred_rf_bow = rf_classifier_bow.predict(x_test_bow)
y_pred_nb_bow = nb_classifier_bow.predict(x_test_bow)
y_pred_knn_bow = knn_classifier_bow.predict(x_test_bow)
y_pred_svm_bow = svm_classifier_bow.predict(x_test_bow)

"""

> **TD_IDF**

"""

# Decision Trees
dt_classifier_tdidf = DecisionTreeClassifier(random_state=42)
dt_classifier_tdidf.fit(x_train_tfidf, y_train)

# Random Forest
rf_classifier_tdidf = RandomForestClassifier(random_state=42)
rf_classifier_tdidf.fit(x_train_tfidf, y_train)

# Naive Bayes
nb_classifier_tdidf = MultinomialNB()
nb_classifier_tdidf.fit(x_train_tfidf, y_train)

# K-NN Classifier
knn_classifier_tdidf = KNeighborsClassifier(n_neighbors=5)
knn_classifier_tdidf.fit(x_train_tfidf, y_train)

# SVM Classifier
svm_classifier_tdidf = SVC(probability=True, random_state=42)
svm_classifier_tdidf.fit(x_train_tfidf, y_train)

"""**TD_IDF Prediction**"""

y_pred_dt_tdidf = dt_classifier_tdidf.predict(x_test_bow)
y_pred_rf_tdidf = rf_classifier_tdidf.predict(x_test_bow)
y_pred_nb_tdidf = nb_classifier_tdidf.predict(x_test_tfidf)
y_pred_knn_tdidf = knn_classifier_tdidf.predict(x_test_bow)
y_pred_svm_tdidf = svm_classifier_tdidf.predict(x_test_tfidf)

"""**Definig a Function to calculate Metrics**"""

def print_metrics(y_true, y_pred):
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_true, y_pred))

"""**Metrics for Bag_of_Words Prediction**"""

# Decision Trees
print("Decision Trees:")
print_metrics(y_test, y_pred_dt_bow)

# Random Forest
print("\nRandom Forest:")
print_metrics(y_test, y_pred_rf_bow)

# Naive Bayes
print("\nNaive Bayes:")
print_metrics(y_test, y_pred_nb_bow)

# K-NN Classifier
print("\nK-NN Classifier:")
print_metrics(y_test, y_pred_knn_bow)

# SVM Classifier
print("\nSVM Classifier:")
print_metrics(y_test, y_pred_svm_bow)

"""**Metrics For TD_IDF Prediction**"""

# Decision Trees
print("Decision Trees:")
print_metrics(y_test, y_pred_dt_tdidf)

# Random Forest
print("\nRandom Forest:")
print_metrics(y_test, y_pred_rf_tdidf)

# Naive Bayes
print("\nNaive Bayes:")
print_metrics(y_test, y_pred_nb_tdidf)

# K-NN Classifier
print("\nK-NN Classifier:")
print_metrics(y_test, y_pred_knn_tdidf)

# SVM Classifier
print("\nSVM Classifier:")
print_metrics(y_test, y_pred_svm_tdidf)
