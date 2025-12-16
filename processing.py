import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


#thiss function loadds amazon dataset
def load_dataset_amazon(train_path, test_path, max_train_rows=None, max_test_rows=None):
    print("Loading train CSV")
    train_data = pd.read_csv(train_path, header=None, names=["rating", "title", "review"])
    print("Loading test CSV")
    test_data = pd.read_csv(test_path, header=None, names=["rating", "title", "review"])

    #combine title and review
    train_data["full"] = train_data["title"].fillna("") + " " + train_data["review"].fillna("")
    test_data["full"] = test_data["title"].fillna("") + " " + test_data["review"].fillna("")


    #map ratings to 0(negative) andd 1(positive)
    train_data["label"] = train_data["rating"] - 1
    test_data["label"]  = test_data["rating"] - 1

    #subsample used for testing
    if max_train_rows is not None and len(train_data) > max_train_rows:
        train_data = train_data.sample(n=max_train_rows, random_state=0).reset_index(drop=True)
    if max_test_rows is not None and len(test_data) > max_test_rows:
        test_data = test_data.sample(n=max_test_rows, random_state=0).reset_index(drop=True)

    print(f"Train rows after filtering/subsampling: {len(train_data)}")
    print(f"Test rows after filtering/subsampling:  {len(test_data)}")


    #convert to lists
    #used for validation
    train_docs_all = train_data["full"].tolist()
    train_labels_all = train_data["label"].tolist()

    #used for final eval
    test_docs = test_data["full"].tolist()
    test_labels = test_data["label"].tolist()

    return train_docs_all, train_labels_all, test_docs, test_labels


def load_dataset_ag_news(train_path, test_path, max_train_rows=None, max_test_rows=None):
    train_data = pd.read_csv(train_path, header=None, names=["class", "title", "desc"])
    test_data = pd.read_csv(test_path, header=None, names=["class", "title", "desc"])

    train_data["text"] = train_data["title"].fillna("").astype(str) + " " + train_data["desc"].fillna("").astype(str)
    test_data["text"] = test_data["title"].fillna("").astype(str) + " " + test_data["desc"].fillna("").astype(str)


    #subsample used for testing
    if max_train_rows is not None and len(train_data) > max_train_rows:
        train_data = train_data.sample(n=max_train_rows, random_state=0).reset_index(drop=True)

    if max_test_rows is not None and len(test_data) > max_test_rows:
        test_data = test_data.sample(n=max_test_rows, random_state=0).reset_index(drop=True)

    train_data["label"] = (train_data["class"].astype(int) - 1)
    test_data["label"] = (test_data["class"].astype(int) - 1)

    train_docs = train_data["text"].tolist()
    train_labels = train_data["label"].tolist()
    test_docs = test_data["text"].tolist()
    test_labels = test_data["label"].tolist()

    return train_docs, train_labels, test_docs, test_labels

#this function makes unlabeledd data pool for LTS for querying LLM
#also makes a validation set for evaluating for f1
def pool_val(train_docs_all, train_labels_all, val_size=0.1, random_state=0):
    #make pool and validation
    train_docs, val_docs, _, val_labels = train_test_split(train_docs_all, train_labels_all, test_size=0.1, random_state=0, stratify=train_labels_all)

    return train_docs, val_docs, val_labels

#tfiddf converts to vectors including frequency of wordddds and rarity across dataset
def make_tfidf(train_docs, val_docs, test_docs, max_features=20_000):
    print("Fitting TF-IDF vectorizer")

    #init tfidif vectorizer
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2)
    vectorizer.fit(train_docs + val_docs)

    #convert to tfidf feature vectors
    #unlabelled training poool
    X_pool = vectorizer.transform(train_docs)

    #validation docs
    X_val = vectorizer.transform(val_docs)

    #test ddocs
    X_test = vectorizer.transform(test_docs)

    return vectorizer, X_pool, X_val, X_test