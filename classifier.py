from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


#train classifier using logistc regression on labeled docs to predict sentiment
def train_classifier(X_labeled, y_labeled):
    #init classifier
    #"balanced" helps compensate for any class imbalance
    classifier = LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=-1)

    #learn which words are positive andd negative using the tfidf vectors and labels from llm
    classifier.fit(X_labeled, y_labeled)

    return classifier

#compute f1 score to eval performance
def compute_f1(classifier, X, y_true):
    #predict labels
    preds = classifier.predict(X)
    #return f1
    #"macro" helps when labels are imbalanced
    #zero division prevents crashing if case where classifier predicts only one class
    return f1_score(y_true, preds, average="macro", zero_division=0)
