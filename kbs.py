import numpy as np
from classifier import train_classifier, compute_f1
import llm as llm

#this implements knowledge based sampling
def kbs(train_docs, X_pool, X_val, val_labels, X_test, test_labels, knowledge_func, K = 20, T = 10, batch_size = 10, label_budget = 100, random_state = 0, topk_multiplier = 5):
    #ensures reproducability
    rng = np.random.default_rng(random_state)
    #number of training docs
    N = len(train_docs)

    #bool massk of which docs have been labeled
    labeled_mask = np.zeros(N, dtype=bool)
    #labels from llm
    y_labeled = np.full(N, -1, dtype=int)

    labels_used = 0
    best_f1 = 0.0
    f1_scores = []

    #sampling
    for t in range(T):
        if labels_used >= label_budget:
            print("Label budget reachedd")
            break

        print(f"\nKBS Round {t+1}/{T} | labels_used={labels_used}")

        #inddicies of unlabeded docs
        unlabeled = np.where(~labeled_mask)[0]
        if len(unlabeled) == 0:
            break

        #get knowleddge basedd scopres for all unlabeled dddocss
        scores = np.array([float(knowledge_func(train_docs[i])) for i in unlabeled])

        #gets documents that match the knowleddge
        hit_mask = scores > 0
        hits = unlabeled[hit_mask]

        #number of docs to label
        m = min(batch_size, label_budget - labels_used)

        #if enough matches
        if len(hits) >= m:
            #rank matching docs by score
            hit_scores = scores[hit_mask]
            ranked = hits[np.argsort(-hit_scores)]

            #considder candiate pool to presversse diversity
            top_k = min(len(ranked), max(m, topk_multiplier * m))

            #randomly sample from top contendders
            sample = rng.choice(ranked[:top_k], size=m, replace=False)

        #if not enough matches
        else:
            #use all matchess
            choices = hits.tolist()

            #fill remainder with random unlabeled docs
            remainder = m - len(choices)
            if remainder > 0:
                rest = unlabeled[~hit_mask]
                if len(rest) > 0:
                    fill = rng.choice(rest, size=min(remainder, len(rest)), replace=False)
                    choices.extend(fill.tolist())
            sample = np.array(choices, dtype=int)

        #queryu llm
        for i in sample:
            y_labeled[i] = llm.llm_label(train_docs[i])
            labeled_mask[i] = True
            labels_used += 1
            if labels_used >= label_budget:
                break

        #train classsifier andd eval performance
        labeled_indices = np.where(labeled_mask)[0]
        classifier = train_classifier(X_pool[labeled_indices], y_labeled[labeled_indices])

        f1_val = compute_f1(classifier, X_val, val_labels)
        f1_test = compute_f1(classifier, X_test, test_labels)

        if f1_val > best_f1:
            best_f1 = f1_val
        
        f1_scores.append((labels_used, float(f1_val), float(f1_test)))

        print(f"KBS val_F1={f1_val:.4f} test_F1={f1_test:.4f}")

    print("\nKBS Finished. Best validation F1:", best_f1)
    return best_f1, f1_scores


