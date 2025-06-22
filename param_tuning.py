import optuna
from optuna.samplers import TPESampler
from itertools import product
import random
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, calinski_harabasz_score, adjusted_rand_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import ParameterGrid
from nltk.tokenize import word_tokenize
from collections import Counter
import importlib.util
from main import seed_words, full_training_loop
from ours import purity_score_filtered
from gensim import corpora
from nltk.corpus import stopwords
import random
np.random.seed(42)
random.seed(42)

# === Load data ===
# Load synthetic dataset
df = pd.read_csv("path to the training data")
def load_stopwords(file_path):
    with open(file_path, encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())
def preprocess_text(text, custom_stopwords):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in custom_stopwords]
    return tokens
custom_stopwords = load_stopwords('stopwords-fi.txt')
documents = df['Sentence'].astype(str)
true_labels = df['Label'].tolist()
processed_docs = documents.apply(lambda x: preprocess_text(x, custom_stopwords))
dictionary = corpora.Dictionary(processed_docs)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
documents = processed_docs.apply(lambda x: ' '.join(x))






# Label preprocessing
y_true = np.array(true_labels)
unique_labels = np.unique(y_true[y_true != '-1'])
# Create a mapping for labels, starting from 1
label_mapping = {label: idx for idx, label in enumerate(unique_labels, start=1)}
ytrue_example = np.array([label_mapping.get(label, -1) for label in y_true])
valid_idx = np.where(ytrue_example != -1)[0]
filtered_ytrue = ytrue_example[valid_idx]

# TF-IDF
tfidf_vectorizer = TfidfVectorizer()
V = tfidf_vectorizer.fit_transform(documents)
feature_names = tfidf_vectorizer.get_feature_names_out()
MH_indices=list(range(15))
# Seed word setup
seed_indices = [i for i, word in enumerate(feature_names) if word in seed_words]
zero_seed_docs = np.where(np.sum(V[:, seed_indices], axis=1) == 0)[0]





# Define the hyperparameter grid
param_grid = {
    "lambda1": [0.01, 0.05, 0.2],
    "lambda2": [0.05, 0.1, 0.3],
    "lambda3": [0.01, 0.05, 0.2],
    "lambda4": [0.1, 0.2, 0.5],
    "lambda_H": [0.01, 0.05, 0.2],
    "eta_b": [0.005, 0.01, 0.05],
    "eta_a": [0.005, 0.01, 0.05],
    "eta_e": [0.005, 0.01, 0.05],
    "eta_p": [0.1, 0.3, 0.5],
    "sparsity_p": [0.1, 0.3, 0.5]
}

param_combinations = list(product(*param_grid.values()))
random.shuffle(param_combinations)  # optional, for diversity
results_clustering = []
results_topic=[]
# Loop through parameter configs
for idx, combo in enumerate(param_combinations[:10]):  # limit for testing
    params = dict(zip(param_grid.keys(), combo))
    print(f"\n🔧 Run {idx+1} with params: {params}")

    # Run training
    W_combined, H_combined, b, a, e, p, losses = full_training_loop(
        V,
        n_iter=35,
        k=50,
        C=18,
        MH_indices=MH_indices,
        I_0=zero_seed_docs,
        seed_indices=seed_indices,
        **params
    )
    
    # Compute metrics for clustering
    labels_int = np.argmax(p, axis=1)
    filtered_labels_int = labels_int[valid_idx]
    sil = silhouette_score(p, labels_int)
    nmi = normalized_mutual_info_score(filtered_ytrue, filtered_labels_int)
    purity = purity_score_filtered(filtered_ytrue, filtered_labels_int, exclude_labels_from_majority=[-1], exclude_labels_from_purity=[])
    ari = adjusted_rand_score(filtered_ytrue, filtered_labels_int)
    ch_score = calinski_harabasz_score(p, labels_int)
    db_score = davies_bouldin_score(p, labels_int)
    entropy_p = -np.sum(p * np.log(p + 1e-12)) / len(p)


    #Compute metrics for topic quality
    labels_int_topic= np.argmax(W_combined, axis=1)
    filtered_labels_int_topic = labels_int_topic[valid_idx]
    nmi_int_topic = normalized_mutual_info_score(filtered_ytrue, filtered_labels_int_topic)
    purity_int_topic = purity_score_filtered(filtered_ytrue, filtered_labels_int_topic, exclude_labels_from_majority=[-1], exclude_labels_from_purity=[])
    ari_int_topic = adjusted_rand_score(filtered_ytrue, filtered_labels_int_topic)


    
    loss_dict = {f"loss_{i+1}": val for i, val in enumerate(losses)}

    # Save result
    result = {
        "run_id": idx + 1,
        **params,
        "silhouette": sil,
        "nmi": nmi,
        "purity": purity,
        "ari": ari,
        "calinski_harabasz": ch_score,
        "davies_bouldin": db_score,
        "entropy_p": entropy_p,
        **loss_dict
    }
    results_clustering.append(result)
    result_topic ={
        "run_id": idx + 1,
        **params,
        "nmi_topic": nmi_int_topic,
        "purity_topic": purity_int_topic,
        "ari_topic": ari_int_topic,
        **loss_dict
    }
    results_topic.append(result_topic)
# Save to CSV
df_results_clustering = pd.DataFrame(results_clustering)
df_results_topic= pd.DataFrame(results_topic)
df_results_clustering.to_csv("parameter_tuning_clustering.csv", index=False)
df_results_topic.to_csv("parameter_tuning_topic_modeling.csv", index=False)
