import numpy as np
import pandas as pd
import llm as llm
import processing as processing
from classifier import train_classifier, compute_f1

#this implements random sampling
def random_sampling(train_docs, X_pool, X_val, val_labels, X_test, test_labels, T = 10, batch_size = 10, label_budget = 100, random_state = 0):
    #ensures reproducability
    rng = np.random.default_rng(random_state)
    val_labels = np.array(val_labels, dtype=int)
    test_labels = np.array(test_labels, dtype=int)

    #number of traiing docs
    N = len(train_docs)

    #shows which docs have been labeld
    labeled_mask = np.zeros(N, dtype=bool)

    #llm labelss
    y_labeled = np.full(N, -1, dtype=int)

    labels_used = 0
    best_f1 = 0.0
    f1_scores = []

    for t in range(T):
        if labels_used >= label_budget:
            print("Label budget reached")
            break

        print(f"\nRANDOM Round {t+1}/{T} | labels_used={labels_used}")

        #currently unlabeldd dddocs
        unlabeled = np.where(~labeled_mask)[0]
        if len(unlabeled) == 0:
            print("No unlabeled docs left")
            break
        
        #number fo ddocs to label
        m = min(batch_size, len(unlabeled), label_budget - labels_used)

        #randomly sample unlabbeld docs
        sample = rng.choice(unlabeled, size=m, replace=False)

        #query llm for labels
        for i in sample:
            label_hat = llm.llm_label(train_docs[i])
            y_labeled[i] = label_hat
            labeled_mask[i] = True
            labels_used += 1

        #get labeledd docs
        labeled_indices = np.where(labeled_mask)[0]
        X_labeled = X_pool[labeled_indices]
        y_labeled_used = y_labeled[labeled_indices]

        #train classifier
        classifier = train_classifier(X_labeled, y_labeled_used)
        f1_val = compute_f1(classifier, X_val, val_labels)
        f1_test = compute_f1(classifier, X_test, test_labels)

        if f1_val > best_f1:
            best_f1 = f1_val

        f1_scores.append((labels_used, float(f1_val), float(f1_test)))

        print(f"RANDOM val_F1={f1_val:.4f} test_F1={f1_test:.4f}")

    print("\nRANDOM Finished. Best validation F1:", best_f1)
    return best_f1, f1_scores