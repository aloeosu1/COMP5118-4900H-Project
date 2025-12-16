import processing as processing
import lts as lts
import random_sampling as random_sampling
import knowledge as knowledge
import kbs as kbs


#main   
if __name__ == "__main__":
    random_state = 0

    #amazon ddataset path
    #train_path = "C:/Users/micha/Desktop/Fall2025/COMP5108/Project/datasets/amazon_review_polarity_csv/train.csv"
    #test_path = "C:/Users/micha/Desktop/Fall2025/COMP5108/Project/datasets/amazon_review_polarity_csv/test.csv"

    #ag news dataset path
    train_path = "C:/Users/micha/Desktop/Fall2025/COMP5108/Project/datasets/ag_news_csv/train.csv"
    test_path = "C:/Users/micha/Desktop/Fall2025/COMP5108/Project/datasets/ag_news_csv/test.csv"


    #load and process data amazon
    #train_docs_all, train_labels_all, test_docs, test_labels = processing.load_dataset_amazon(train_path, test_path, max_train_rows=20_000, max_test_rows=5_000)

    #loadd and processs ag newss
    train_docs_all, train_labels_all, test_docs, test_labels = processing.load_dataset_ag_news(train_path, test_path, max_train_rows=20_000, max_test_rows=5_000)


    #make pool and validdation
    train_docs, val_docs, val_labels = processing.pool_val(train_docs_all, train_labels_all, val_size=0.1, random_state=random_state)

    #tfiddf features
    _, X_pool, X_val, X_test = processing.make_tfidf(train_docs, val_docs, test_docs)

    #LTS test
    K = 20
    rounds = 10
    batch_size = 10
    label_budget = 50   #relatedd to real money

    lts_best_f1, lts_f1_scores = lts.learn_to_sample(
        train_docs,
        X_pool,
        X_val,
        val_labels,
        X_test,
        test_labels,
        K=K,
        T=rounds,
        batch_size=batch_size,
        label_budget=label_budget,
        random_state=random_state,
    )

    print("\nLTS F1 scores (labels_used, val_F1, test_F1):")
    for step in lts_f1_scores:
        print(step)


    #randdom test
    random_best_f1, random_f1_scores = random_sampling.random_sampling(
        train_docs,
        X_pool,
        X_val,
        val_labels,
        X_test,
        test_labels,
        T=rounds,
        batch_size=batch_size,
        label_budget=label_budget,
        random_state=random_state,
    )

    print("\nRandom F1 scoress (labels_used, val_F1, test_F1):")
    for step in random_f1_scores:
        print(step)

    #kbs test
    kbs_best_f1, kbs_f1_scores = kbs.kbs(
        train_docs,
        X_pool,
        X_val,
        val_labels,
        X_test,
        test_labels,
        knowledge_func=knowledge.amazon_knowledge,
        K=K,
        T=rounds,
        batch_size=batch_size,
        label_budget=label_budget,
        random_state=random_state,
    )

    print("\nKBS F1 scoress (labels_used, val_F1, test_F1):")
    for step in kbs_f1_scores:
        print(step)

    #final f1 scores
    print("\nLearn to Sample Best F1: ", lts_best_f1)
    print("Random sampling Best F1: ", random_best_f1)
    print("Knowledge based sampling Best F1: ", kbs_best_f1)

