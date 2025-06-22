import pandas as pd
import numpy as np
import random
import re
import string
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, calinski_harabasz_score, adjusted_rand_score, davies_bouldin_score, adjusted_mutual_info_score
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.mixture import GaussianMixture
from collections import Counter, defaultdict
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
#from main import seed_words
from main01 import full_training_loop
from ours import purity_score_filtered
from OurAlgorithm import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.manifold import TSNE
import time
from scipy.optimize import linear_sum_assignment

import scipy.sparse as sp
import time
from numpy.linalg import norm
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import issparse











def normalized_mutual_information(y_true, y_pred):
    """
    Compute Normalized Mutual Information (NMI) between ground truth labels y_true and predicted labels y_pred
    following the exact definition provided.
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Total number of samples
    N = len(y_true)

    # Compute counts
    true_counter = Counter(y_true)
    pred_counter = Counter(y_pred)
    joint_counter = Counter(zip(y_true, y_pred))

    # Compute probabilities
    p_true = {label: count / N for label, count in true_counter.items()}
    p_pred = {label: count / N for label, count in pred_counter.items()}
    p_joint = {labels: count / N for labels, count in joint_counter.items()}

    # Compute Mutual Information MI(C, C0)
    MI = 0.0
    for (ci, cj), p_cicj in p_joint.items():
        if p_cicj > 0:
            MI += p_cicj * np.log2(p_cicj / (p_true[ci] * p_pred[cj]))

    # Compute entropies H(C) and H(C0)
    H_true = -sum(p * np.log2(p) for p in p_true.values() if p > 0)
    H_pred = -sum(p * np.log2(p) for p in p_pred.values() if p > 0)

    # Compute NMI
    NMI = MI / max(H_true, H_pred)

    return NMI

"""
    Compute clustering accuracy (ACC) following the exact definition:
    ACC = (sum of correctly mapped labels) / (total samples)
    where mapping is found via the Hungarian algorithm.
"""
def clustering_accuracy(y_true, y_pred):

    # Corrected: use maximum label value, not number of unique labels
    num_classes = max(y_true.max(), y_pred.max()) + 1
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for true_label, pred_label in zip(y_true, y_pred):
        confusion_matrix[pred_label, true_label] += 1

    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
    total_correct = confusion_matrix[row_ind, col_ind].sum()
    acc = total_correct / y_true.shape[0]

    return acc






seed_words = [
    # Agriculture / Grain (already covered)
    'wheat', 'corn', 'barley', 'oats', 'soybean',
    'harvest', 'crop', 'bushel', 'yield', 'export',
    'import', 'commodity', 'market', 'futures', 'agriculture',
    'grain', 'farmer', 'planting', 'seeding', 'harvesting',
    'irrigation', 'pesticide', 'fertilizer', 'drought', 'rainfall',
    'storage', 'elevator', 'shipment', 'price', 'supply',
    'demand', 'acreage', 'production', 'consumption', 'distribution',
    'trading', 'broker', 'exchange', 'stocks', 'inventory',
    'weather', 'climate', 'forecast', 'plant', 'sow',
    'reap', 'soil', 'moisture', 'disease', 'pest',
    'livestock', 'hog', 'cattle', 'dairy', 'meat',
    'protein', 'biofuel', 'ethanol', 'biodiesel', 'processing',
    'mill', 'flour', 'animal', 'feed', 'residue',
    'rotation', 'paddock', 'pasture', 'grower', 'cultivation',
    'organic', 'conventional', 'resistant', 'hybrid', 'seed',
    'germination', 'commodity_price', 'cash_crop', 'multicropping',
    'subsidy', 'tariff', 'quota', 'negotiation', 'shipment_cost',
    'fertilizer_cost', 'equipment', 'machinery', 'plant_disease', 'agronomy',
    'policy', 'bioengineering', 'genetically_modified', 'cropland', 'output',
    'import_tariff', 'export_tariff', 'supply_chain', 'distribution_center', 'storage_facility',

    # ➡️ Additional for "earn" (finance, companies, earnings)
    'profit', 'revenue', 'dividend', 'share', 'shareholder',
    'quarterly', 'report', 'finance', 'financial', 'stock_market',
    'investment', 'investor', 'fund', 'capital', 'merger',
    'growth', 'assets', 'liability', 'balance_sheet', 'valuation',

    # ➡️ Additional for "acq" (acquisition, mergers)
    'merger', 'acquisition', 'buyout', 'deal', 'ownership',
    'stake', 'partner', 'transaction', 'corporate', 'takeover',
    'agreement', 'negotiation', 'sell', 'purchase', 'combine',

    # ➡️ Additional for "crude" (oil, petroleum)
    'crude', 'oil', 'barrel', 'petroleum', 'refinery',
    'fuel', 'energy', 'gasoline', 'diesel', 'oil_price',
    'opec', 'drilling', 'pipeline', 'exploration', 'fossil_fuel',

    # ➡️ Additional for "money-fx" (currency exchange, forex)
    'currency', 'foreign_exchange', 'forex', 'usd', 'euro',
    'yen', 'pound', 'dollar', 'devaluation', 'exchange_rate',
    'market_rate', 'inflation', 'deflation', 'interest_rate', 'central_bank'
]


def clean_english_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove HTML tags, URLs, and markdown
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
    
    # Remove lines that look like tables (heuristic: many numbers/symbols or tabs)
    text = re.sub(r'[\t]+|[\d\W]{5,}', ' ', text)
    
    # Remove punctuation and digits
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and short tokens
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    
    return ' '.join(tokens)

start_time = time.time()


def preprocess_texti(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens


def load_stopwords(file_path):
    with open(file_path, encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())
def preprocess_text(text, custom_stopwords):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in custom_stopwords]
    return tokens
#custom_stopwords = load_stopwords('/Users/smn573/Documents/constrained_icml_experiment/stopwords-fi.txt')
#custom_stopwords= load_stopwords("stopwords-fi.txt")



#data = pd.read_csv('./synthetic-data.csv')
#data= pd.read_csv('/Users/smn573/Documents/constrained_icml_experiment/synthetic-data.csv')

data= pd.read_csv('20newsgroup_reuters/mixed_reuters_20ng_synthetic_style_org.csv')
#data= pd.read_csv("20newsgroup_reuters/mixed_reuters_20ng_synthetic_style_1000.csv")
documents = data['Sentence'].astype(str)
true_labels = data['Label'].tolist()
#documents = data["Comments"].astype(str)
#processed_doc = documents.apply(clean_english_text)
processed_docs = documents.apply(preprocess_texti)

#processed_docs = documents.apply(lambda x: preprocess_text(x, custom_stopwords))


dictionary = corpora.Dictionary(processed_docs)

corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

documents = processed_docs.apply(lambda x: ' '.join(x))
#tfidf_vectorizer = TfidfVectorizer()
#tfidf_vectorizer = TfidfVectorizer(min_df=1, max_df=98)
tfidf_vectorizer = TfidfVectorizer()
V = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()


seed_indices = [i for i, word in enumerate(tfidf_feature_names) if word in set(seed_words)]
non_seed_indices = [i for i in range(len(tfidf_feature_names)) if i not in seed_indices]
print("number of seed words in vocab: %d" % len(seed_indices))

doc_seedword_sums = np.sum(V[:, seed_indices], axis=1)
zero_seedword_indices = np.where(doc_seedword_sums == 0)[0]
print("zero seed containing:", len(zero_seedword_indices))
print("All seed words: %d", len(set(seed_words)))
num_documents = len(data["Sentence"])
print("Number of documents:", num_documents)
unique_words_count = len(dictionary)
print("Number of unique words:", unique_words_count)
# data['word_count'] = data["Sentence"].apply(lambda x: len(x.split()))
# avg_word_count = data['word_count'].mean()
# print("Average word count per document:", avg_word_count)








ytrue_example = np.array(true_labels)

valid_idx = np.where(ytrue_example != -1)[0]
filtered_ytrue = ytrue_example[valid_idx]
mh_mask = ytrue_example != -1



def top_words_per_topic(H, feature_names, top_n=10):
    print("\nTop words per topic second:")
    for k, topic in enumerate(H):
        top_indices = topic.argsort()[-top_n:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        print(f"Topic {k}: {', '.join(top_words)}")
##############################################################################
MH_indices=list(range(10))
k_cluster=10 #number of the clusters
n_topics=50
W_max=1e-9
theta_min=0.4
all_results = []


for seed in [42, 1122, 7, 99, 2025, 1234]:
    np.random.seed(seed)
    random.seed(seed)
    W_sep, H_sep , kl_losses_sep= train(V, n_topics, MH_indices, W_max, non_seed_indices,seed_indices, theta_min)

    W_combined, H_combined, b, a, e, p, losses = full_training_loop(
        V,
        n_iter=20,
        lambda1=0.2,
        lambda2=0.1,
        lambda3=0.2,
        lambda4=0.2,
        eta_b=0.005,
        eta_a=0.001,
        eta_e=0.005,
        eta_p=0.3,
        k=n_topics,
        C=k_cluster,
        MH_indices=MH_indices,
        I_0=zero_seedword_indices,
        seed_indices=seed_indices,
        lambda_H=0.05,
        sparsity_p=0.3)


    print("Our Constraint Model Topic Words:  \n ")
    top_words_per_topic(H_sep, tfidf_feature_names, top_n=10)

    # # print("-------------------------------------------------------------------------------------")
    print(" \n Integrated Model Topic Words: ")
    top_words_per_topic(H_combined, tfidf_feature_names, top_n=10)



    #X_sep_non_normalized= np.load("W_sep.npy")
    X_sep_normalized=W_sep
    #X_sep_normalized = normalize(X_sep_non_normalized)
    #X_sep_normalized = normalize(X_sep_non_normalized, norm="l1", axis=1) 
    clustering_sep = SpectralClustering(n_clusters=k_cluster)
    #clustering_sep = KMeans(n_clusters=k_cluster)
    #clustering_sep= GaussianMixture(n_components=k_cluster)
    labels_sep = clustering_sep.fit_predict(X_sep_normalized)

    filtered_labels_sep = labels_sep[valid_idx]
    #print("sep: ", filtered_labels_sep)
    sil_sep = silhouette_score(X_sep_normalized, labels_sep)
    ch_sep = calinski_harabasz_score(X_sep_normalized, labels_sep)
    devious_sep = davies_bouldin_score(X_sep_normalized, labels_sep)
    nmi_sep = normalized_mutual_info_score(filtered_ytrue, filtered_labels_sep)
    purity_sep = purity_score_filtered(filtered_ytrue, filtered_labels_sep, exclude_labels_from_majority=[], exclude_labels_from_purity=[])
    ari_sep = adjusted_rand_score(filtered_ytrue, filtered_labels_sep)
    clustering_ac_sep= clustering_accuracy(filtered_ytrue, filtered_labels_sep)

    """
    # === Integrated Model Evaluation ===
    cluster_cols = [col for col in p_df.columns if col.startswith("Cluster_")]
    X_int_non_normalized = p_df[cluster_cols].values
    """

    #print(X_int_non_normalized)
    #print("before normalization", X_int_non_normalized.sum(axis=1))
    #X_int = normalize(MinMaxScaler().fit_transform(X_int_non_normalized))
    #print("after normalization", X_int.sum(axis=1))


    #X_int_non_normalized= np.load("P.npy")
    X_int_non_normalized=p
    #X_int_non_normalized = normalize(X_int_non_normalized, norm="l1", axis=1) 
    labels_int = np.argmax(X_int_non_normalized, axis=1)
    filtered_labels_int = labels_int[valid_idx]
    sil_int = silhouette_score(X_int_non_normalized, labels_int)
    ch_int = calinski_harabasz_score(X_int_non_normalized, labels_int)
    devious_int = davies_bouldin_score(X_int_non_normalized, labels_int)
    nmi_int = normalized_mutual_info_score(filtered_ytrue, filtered_labels_int)
    purity_int = purity_score_filtered(filtered_ytrue, filtered_labels_int, exclude_labels_from_majority=[], exclude_labels_from_purity=[])
    ari_int = adjusted_rand_score(filtered_ytrue, filtered_labels_int)
    clustering_ac_int= clustering_accuracy(filtered_ytrue, filtered_labels_int)
    # === Report ===
    print("\n=== 📊 Document-Level Clustering Comparison ===")
    print("🔹 Separate Model (Topic → Clustering):")
    print(f"  - Accuracy:           {clustering_ac_sep:.4f}")
    print(f"  - Silhouette:         {sil_sep:.4f}")
    print(f"  - NMI:                {nmi_sep:.4f}")
    print(f"  - Adjusted Rand:      {ari_sep:.4f}")
    print(f"  - Calinski-Harabasz:  {ch_sep:.2f}")
    print(f"  - Purity:             {purity_sep:.4f}")
    print(f"  - davies_bouldin:     {devious_sep:.4f}")


    print("\n🔸 Integrated Model (Joint Topic + Clustering):")
    print(f"  - Accuracy:           {clustering_ac_int:.4f}")
    print(f"  - Silhouette:         {sil_int:.4f}")
    print(f"  - NMI:                {nmi_int:.4f}")
    print(f"  - Adjusted Rand:      {ari_int:.4f}")
    print(f"  - Calinski-Harabasz:  {ch_int:.2f}")
    print(f"  - Purity:             {purity_int:.4f}")
    print(f"  - davies_bouldin:     {devious_int:.4f}")
    #print(np.bincount(p.argmax(axis=1)))
    #NMI and Ourity Scores for the topic modeling
    ###########################################################################

    labels_sep_topic= np.argmax(W_sep, axis=1)
    filtered_labels_sep_topic = labels_sep_topic[valid_idx]
    purity_sep_topic = purity_score_filtered(filtered_ytrue, filtered_labels_sep_topic, exclude_labels_from_majority=[], exclude_labels_from_purity=[])
    nmi_sep_topic = normalized_mutual_info_score(filtered_ytrue, filtered_labels_sep_topic)
    ari_sep_topic= adjusted_rand_score(filtered_ytrue, filtered_labels_sep_topic)
    acc_sep_topic= clustering_accuracy(filtered_ytrue, filtered_labels_sep_topic)
    print("\n=== 📊 Document-Level Topic Quality Comparison ===")
    print("🔹 Separate Topic Model (Separate Model):")
    print(f"  - Accuracy Topics:    {acc_sep_topic:.4f}")
    print(f"  - NMI:                {nmi_sep_topic:.4f}")
    print(f"  - Purity:             {purity_sep_topic:.4f}")
    #print(f"  - Adjusted Rand:      {ari_sep_topic:.4f}")

    labels_int_topic= np.argmax(W_combined, axis=1)
    filtered_labels_int_topic = labels_int_topic[valid_idx]
    purity_int_topic = purity_score_filtered(filtered_ytrue, filtered_labels_int_topic, exclude_labels_from_majority=[], exclude_labels_from_purity=[])
    nmi_int_topic = normalized_mutual_info_score(filtered_ytrue, filtered_labels_int_topic)
    ari_int_topic = adjusted_rand_score(filtered_ytrue, filtered_labels_int_topic)
    acc_int_topic= clustering_accuracy(filtered_ytrue, filtered_labels_int_topic)
    print("\n🔸 Integrated Topic Model (Joint Model):")
    print(f"  - Accuracy Topics:    {acc_int_topic:.4f}")
    print(f"  - NMI:                {nmi_int_topic:.4f}")
    print(f"  - Purity:             {purity_int_topic:.4f}")
    #print(f"  - Adjusted Rand:      {ari_int_topic:.4f}")


# 🧪 Then compute all your evaluation metrics:
    metrics = {
        "seed": seed,
        "Separate_Accuracy": clustering_ac_sep,
        "Integrated_Accuracy": clustering_ac_int,
        "Separate_NMI": nmi_sep,
        "Integrated_NMI": nmi_int,
        "Separate_Purity": purity_sep,
        "Integrated_Purity": purity_int,
        "Separate_Silhouette": sil_sep,
        "Integrated_Silhouette": sil_int,
        "Separate_Calinski": ch_sep,
        "Integrated_Calinski": ch_int,
        "Separate_Davies": devious_sep,
        "Integrated_Davies": devious_int,



        "Separate_Topic_Accuracy": acc_sep_topic,
        "Integrated_Topic_Accuracy": acc_int_topic,
        "Separate_Topic_NMI": nmi_sep_topic,
        "Integrated_Topic_NMI": nmi_int_topic,
        "Separate_Topic_Purity": purity_sep_topic,
        "Integrated_Topic_Purity": purity_int_topic
    }

    all_results.append(metrics)

# Save all raw results
df = pd.DataFrame(all_results)
df.to_csv("20newsgroup_reuters/all_seed_results.csv", index=False)




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

# Load your file
df = pd.read_csv("20newsgroup_reuters/all_seed_results.csv")

# Extract all metric names automatically
metric_names = sorted(set(col.replace("Separate_", "").replace("Integrated_", "") for col in df.columns if col.startswith("Separate_")))

# Prepare results
ttest_results = []

for metric in metric_names:
    separate_col = f"Separate_{metric}"
    integrated_col = f"Integrated_{metric}"
    
    if separate_col not in df.columns or integrated_col not in df.columns:
        print(f"Skipping {metric}: missing columns.")
        continue
    
    separate_scores = df[separate_col]
    integrated_scores = df[integrated_col]
    
    # Special handling: if lower is better (Davies index)
    if "Davies" in metric or "davies" in metric:
        separate_scores, integrated_scores = integrated_scores, separate_scores
    
    t_stat, p_val = ttest_rel(separate_scores, integrated_scores)
    mean_separate = separate_scores.mean()
    mean_integrated = integrated_scores.mean()
    
    ttest_results.append({
        "Metric": metric,
        "Separate Mean": mean_separate,
        "Integrated Mean": mean_integrated,
        "T-Statistic": t_stat,
        "P-Value": p_val,
        "Winner": "Integrated" if mean_integrated > mean_separate else "Separate"
    })

# Convert to DataFrame
ttest_df = pd.DataFrame(ttest_results)

# Save to CSV
ttest_df.to_csv("20newsgroup_reuters/paired_ttest_results.csv", index=False)
print("✅ Paired t-test results saved to 'paired_ttest_results.csv'.")

# Plot comparison for each metric
for metric in metric_names:
    separate_col = f"Separate_{metric}"
    integrated_col = f"Integrated_{metric}"
    
    if separate_col not in df.columns or integrated_col not in df.columns:
        continue
    
    plt.figure(figsize=(6,4))
    plt.plot(df[separate_col], label="Separate", marker='o')
    plt.plot(df[integrated_col], label="Integrated", marker='x')
    plt.title(f"Scores across Seeds: {metric}")
    plt.xlabel("Run (seed)")
    plt.ylabel(f"{metric} Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"20newsgroup_reuters/comparison_plot_{metric}.png")
    plt.close()

print("✅ Metric comparison plots saved.")
