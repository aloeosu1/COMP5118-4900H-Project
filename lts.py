import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import llm as llm
from classifier import train_classifier, compute_f1

#this function implements the main LTS logic
def learn_to_sample(train_docs, X_pool, X_val, val_labels, X_test, test_labels, K = 20, T = 10, batch_size = 10, label_budget = 100, alpha_prior = 1.0, beta_prior = 1.0, decay = 0.99, random_state = 0):
    #ensures reproducability
    rng = np.random.default_rng(random_state)
    #convert val labels to numpy int array
    val_labels = np.array(val_labels, dtype=int)
    #convert test labels to numpy int array
    test_labels = np.array(test_labels, dtype=int)

    #number of ddocs in unlabeled pool
    N = len(train_docs)

    #clustering the pool (using KMeans instead of topic model)
    print("Cluster with KMeans")
    kmeans = KMeans(n_clusters=K, random_state=random_state, n_init="auto")
    cluster_id = kmeans.fit_predict(X_pool)

    #map cluster id to corresponding doccument indicies
    cluster_to_index = defaultdict(list)
    for i, id in enumerate(cluster_id):
        cluster_to_index[id].append(i)

    
    #label storage for LLM labels
    #shows which docs havbe been labeled
    labeled_mask = np.zeros(N, dtype=bool)
    #labels asssigned by llm
    y_labeled = np.full(N, -1, dtype=int)

    #thompson sampling
    wins = np.zeros(K, dtype=float)
    losses = np.zeros(K, dtype=float)

    #choose cluster
    def select_cluster():
        theta = np.zeros(K)
        for k in range(K):
            alpha = alpha_prior + wins[k]
            beta = beta_prior + losses[k]
            #using beta distribution
            theta[k] = rng.beta(alpha, beta)

        return int(np.argmax(theta))
    

    labels_used = 0
    best_f1 = 0.0
    current_classifier = None
    f1_scores = []

    #for t = 1 to T DO
    for t in range(T):
        if labels_used >= label_budget:
            print("Label budget reached")
            break

        print(f"\nLTS Round {t + 1}/{T} | labels_used={labels_used}")

        #for i = 1 to K DO
        for i in range(K):
            c = select_cluster()
            candidates = [i for i in cluster_to_index[c] if not labeled_mask[i]]
            if candidates:
                cluster = c
                break

        else:
            print("No unlabeled docs in any cluster")
            break

        print(f"Cluster selected: {cluster}")

        #sampling batch from selected cluster
        candidates = [i for i in cluster_to_index[cluster] if not labeled_mask[i]]
        m = min(batch_size, len(candidates), label_budget - labels_used)
        indices = rng.choice(candidates, size=m, replace=False)

        #query LLM for each selected ddoc
        for i in indices:
            text = train_docs[i]
            label_hat = llm.llm_label(text)
            y_labeled[i] = label_hat
            labeled_mask[i] = True
            labels_used += 1

        #debug
        labeled_indices = np.where(labeled_mask)[0]
        print("  LLM label counts so far:", np.bincount(y_labeled[labeled_indices]))

        print(f"Labeled {m} docs this round (T). Total labels_used={labels_used}")

        #train classifer on all LLM-labeled docs
        labeled_indices = np.where(labeled_mask)[0]
        X_labeled = X_pool[labeled_indices]
        y_labeled_used = y_labeled[labeled_indices]

        #if there is only one class so far
        if len(np.unique(y_labeled_used)) < 2:
            print("Only one class so far. Skip training this round.")
            continue

        current_classifier = train_classifier(X_labeled, y_labeled_used)
        f1_val = compute_f1(current_classifier, X_val, val_labels)
        f1_test = compute_f1(current_classifier, X_test, test_labels)

        #updating rewards
        if f1_val > best_f1:
            reward = 1
            best_f1 = f1_val
        
        else:
            reward = 0

        #upddate wins and losses
        if reward > 0:
            wins[cluster] += 1.0
        
        else:
            losses[cluster] += 1.0

        #apply decay
        wins *= decay
        losses *= decay

        #add f1 score to list
        f1_scores.append((labels_used, float(f1_val), float(f1_test)))
    
    #print the best f1 score
    print("\nFinished. Best validation F1:", best_f1)
    return best_f1, f1_scores





