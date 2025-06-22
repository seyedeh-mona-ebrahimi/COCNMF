import pandas as pd
import numpy as np
import random
import re
import string
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, calinski_harabasz_score, adjusted_rand_score, davies_bouldin_score, adjusted_mutual_info_score
from sklearn.cluster import SpectralClustering, KMeans

from sklearn.preprocessing import MaxAbsScaler
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
from scipy.stats import entropy
from matplotlib.ticker import LogFormatterSciNotation
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
np.random.seed(2025)

"""
    Compute clustering accuracy (ACC) following the exact definition:
    ACC = (sum of correctly mapped labels) / (total samples)
    where mapping is found via the Hungarian algorithm.
"""




def clustering_accuracy(y_true, y_pred):
    """
    Computes clustering accuracy using the Hungarian algorithm for optimal assignment.

    Parameters:
    - y_true: array-like of shape (n_samples,) – Ground truth labels
    - y_pred: array-like of shape (n_samples,) – Cluster assignments

    Returns:
    - acc: float – Clustering accuracy
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Mask: exclude majority labels (e.g., -1)
    mask = y_true != -1
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return 0.0  # Avoid division by zero

    D = max(y_pred.max(), y_true.max()) + 1
    conf_mat = np.zeros((D, D), dtype=np.int64)

    for i in range(y_pred.size):
        conf_mat[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(-conf_mat)
    accuracy = conf_mat[row_ind, col_ind].sum() / y_pred.size

    return accuracy




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
    
    #  Additional for "earn" (finance, companies, earnings)
    'profit', 'revenue', 'dividend', 'share', 'shareholder',
    'quarterly', 'report', 'finance', 'financial', 'stock_market',
    'investment', 'investor', 'fund', 'capital', 'merger',
    'growth', 'assets', 'liability', 'balance_sheet', 'valuation',

    #  Additional for "acq" (acquisition, mergers)
    'merger', 'acquisition', 'buyout', 'deal', 'ownership',
    'stake', 'partner', 'transaction', 'corporate', 'takeover',
    'agreement', 'negotiation', 'sell', 'purchase', 'combine',

    #  Additional for "crude" (oil, petroleum)
    'crude', 'oil', 'barrel', 'petroleum', 'refinery',
    'fuel', 'energy', 'gasoline', 'diesel', 'oil_price',
    'opec', 'drilling', 'pipeline', 'exploration', 'fossil_fuel',

    # Additional for "money-fx" (currency exchange, forex)
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
    
    #return ' '.join(tokens)
    return tokens



def preprocess_texti(text):
    # Remove HTML tags, URLs, and markdown
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
    
    # Remove lines that look like tables (heuristic: many numbers/symbols or tabs)
    text = re.sub(r'[\t]+|[\d\W]{5,}', ' ', text)
    
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


data= pd.read_csv('20newsgroup_reuters/mixed_reuters_20ng_synthetic_style_org.csv')
documents = data['Sentence'].astype(str)
true_labels = data['Label'].tolist()

processed_docs = documents.apply(clean_english_text)

dictionary = corpora.Dictionary(processed_docs)

corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

documents = processed_docs.apply(lambda x: ' '.join(x))

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



"""This label mapping is just for Mental health data"""
# y_true = np.array(true_labels)
# unique_labels = np.unique(y_true[y_true != '-1'])
# label_mapping = {label: idx for idx, label in enumerate(unique_labels, start=1)}
# ytrue_example = np.array([label_mapping.get(label, -1) for label in y_true])

"""This label mapping is just for 20Newsgroup and reuters"""
ytrue_example = np.array(true_labels)
valid_idx = np.where(ytrue_example != -1)[0]
filtered_ytrue = ytrue_example[valid_idx]
mh_mask = ytrue_example != -1

##############################################################################
MH_indices=list(range(10))
k_cluster=10 #number of the clusters
n_topics=30
W_max=1e-9
theta_min=0.4
W_sep, H_sep , kl_losses_sep= train(V, n_topics, MH_indices, W_max, non_seed_indices, seed_indices, theta_min)
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

def top_words_per_topic(H, feature_names, top_n=10):
    print("\nTop words per topic second:")
    for k, topic in enumerate(H):
        top_indices = topic.argsort()[-top_n:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        print(f"Topic {k}: {', '.join(top_words)}")

print("The Constraint Model Topic Words (CMTM):  \n ")
top_words_per_topic(H_sep, tfidf_feature_names, top_n=10)

# # print("-------------------------------------------------------------------------------------")
print(" \n Integrated Model Topic Words (COCNMF): ")
top_words_per_topic(H_combined, tfidf_feature_names, top_n=10)

X_sep_normalized=W_sep
clustering_sep = KMeans(n_clusters=k_cluster)
#clustering_sep= GaussianMixture(n_components=k_cluster)
labels_sep = clustering_sep.fit_predict(X_sep_normalized)

filtered_labels_sep = labels_sep[valid_idx]
#print("sep: ", filtered_labels_sep)
sil_sep = silhouette_score(X_sep_normalized, labels_sep)
ch_sep = calinski_harabasz_score(X_sep_normalized, labels_sep)
devious_sep = davies_bouldin_score(X_sep_normalized, labels_sep)
nmi_sep = normalized_mutual_info_score(filtered_ytrue, filtered_labels_sep)
purity_sep = purity_score_filtered(filtered_ytrue, filtered_labels_sep, exclude_labels_from_majority=[-1], exclude_labels_from_purity=[])
ari_sep = adjusted_rand_score(filtered_ytrue, filtered_labels_sep)
clustering_ac_sep= clustering_accuracy(filtered_ytrue, filtered_labels_sep)




#Our Model(COCNMF)
##############################################################################
X_int_non_normalized=p
labels_int = np.argmax(X_int_non_normalized, axis=1)
filtered_labels_int = labels_int[valid_idx]
sil_int = silhouette_score(X_int_non_normalized, labels_int)
ch_int = calinski_harabasz_score(X_int_non_normalized, labels_int)
devious_int = davies_bouldin_score(X_int_non_normalized, labels_int)
nmi_int = normalized_mutual_info_score(filtered_ytrue, filtered_labels_int)
purity_int = purity_score_filtered(filtered_ytrue, filtered_labels_int, exclude_labels_from_majority=[-1], exclude_labels_from_purity=[])
ari_int = adjusted_rand_score(filtered_ytrue, filtered_labels_int)
clustering_ac_int= clustering_accuracy(filtered_ytrue, filtered_labels_int)

# === Report ===
print("\n=== 📊 Document-Level Clustering Comparison ===")
print("🔹 Separate Model (Topic → Clustering):")
print(f"  - NMI:                {nmi_sep:.4f}")
print(f"  - Purity:             {purity_sep:.4f}")
print(f"  - Silhouette:         {sil_sep:.4f}")
print(f"  - Calinski-Harabasz:  {ch_sep:.2f}")
print(f"  - davies_bouldin:     {devious_sep:.4f}")
print(f"  - Accuracy:           {clustering_ac_sep:.4f}")


print("\n")
print("\n🔸 Integrated Model (Joint Topic + Clustering):")
print(f"  - NMI:                {nmi_int:.4f}")
print(f"  - Purity:             {purity_int:.4f}")
print(f"  - Silhouette:         {sil_int:.4f}")
print(f"  - Calinski-Harabasz:  {ch_int:.2f}")
print(f"  - davies_bouldin:     {devious_int:.4f}")
print(f"  - Accuracy:           {clustering_ac_int:.4f}")







#visulaization and comparison with other baselines
#########################################################################
stsne= TSNE(n_components=2, perplexity=40)
sep_tsne= stsne.fit_transform(X_sep_normalized)
itsne= TSNE(n_components=2, perplexity=40)
int_tsne= itsne.fit_transform(p)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)  # list of strings

print("#########################################################################################")
print("\n TOPIC MODELING  →  CLUSTERIN ")


#LDA → k-means
##############################################################################
from gensim.models.ldamodel import LdaModel
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import AgglomerativeClustering


# Train LDA
#lda_model = LdaModel(corpus=corpus, num_topics=n_topics, id2word=dictionary)
lda_model= LdaModel(corpus=corpus, num_topics=n_topics, id2word=dictionary)


lda_doc_topics = np.zeros((len(corpus), n_topics))
for i, doc_bow in enumerate(corpus):
    for topic_id, prob in lda_model.get_document_topics(doc_bow):
        lda_doc_topics[i, topic_id] = prob




#lda_doc_topics = normalize(lda_doc_topics, norm='l2')

# Cluster using k-means
lda_kmeans = KMeans(n_clusters=k_cluster)
labels_lda_kmeans = lda_kmeans.fit_predict(lda_doc_topics)

# Evaluation
filtered_lda_kmeans = labels_lda_kmeans[valid_idx]
print("\n📌 LDA → k-means:")

print(f"- NMI:       {normalized_mutual_info_score(filtered_ytrue, filtered_lda_kmeans):.4f}")
print(f"- Purity:    {purity_score_filtered(filtered_ytrue, filtered_lda_kmeans, exclude_labels_from_majority=[-1]):.4f}")
print(f"- Accuracy:  {clustering_accuracy(filtered_ytrue, filtered_lda_kmeans):.4f}")
print(f"- Silhouette: {silhouette_score(lda_doc_topics, labels_lda_kmeans):.4f}")
print(f"- CH: {calinski_harabasz_score(lda_doc_topics, labels_lda_kmeans):.4f}")
print(f"- DB: {davies_bouldin_score(lda_doc_topics, labels_lda_kmeans):.4f}")



#LDA → HAC
##############################################################################

# Extract topic distributions (theta) for each document
lda_doc_topics = np.zeros((len(corpus), n_topics))
for i, doc_bow in enumerate(corpus):
    for topic_id, prob in lda_model.get_document_topics(doc_bow):
        lda_doc_topics[i, topic_id] = prob


# Step 4: HAC clustering
lda_hac = AgglomerativeClustering(n_clusters=k_cluster)
labels_lda_hac = lda_hac.fit_predict(lda_doc_topics)
# Evaluation
filtered_lda_hac = labels_lda_hac[valid_idx]
print("\n📌 LDA → HAC:")

print(f"- NMI:       {normalized_mutual_info_score(filtered_ytrue, filtered_lda_hac):.4f}")
print(f"- Purity:    {purity_score_filtered(filtered_ytrue, filtered_lda_hac, exclude_labels_from_majority=[-1]):.4f}")
print(f"- Accuracy:  {clustering_accuracy(filtered_ytrue, filtered_lda_hac):.4f}")
print(f"- Silhouette:{silhouette_score(lda_doc_topics, labels_lda_hac):.4f}")
print(f"- CH:        {calinski_harabasz_score(lda_doc_topics, labels_lda_hac):.4f}")
print(f"- DB:        {davies_bouldin_score(lda_doc_topics, labels_lda_hac):.4f}")




# 2. Train NMF (non-negative matrix factorization)
nmf_model = NMF(n_components=n_topics)
nmf_doc_topics = nmf_model.fit_transform(V)  # shape: (n_docs, n_topics)
nmf_doc_topics = normalize(nmf_doc_topics, norm='l2')
# 3. Cluster with KMeans
nmf_kmeans = KMeans(n_clusters=k_cluster)
labels_nmf_kmeans = nmf_kmeans.fit_predict(nmf_doc_topics)

# 4. Evaluate
filtered_nmf_kmeans = labels_nmf_kmeans[valid_idx]

print("\n📌 NMF → k-means:")
print(f"- NMI:       {normalized_mutual_info_score(filtered_ytrue, filtered_nmf_kmeans):.4f}")
print(f"- Purity:    {purity_score_filtered(filtered_ytrue, filtered_nmf_kmeans, exclude_labels_from_majority=[-1]):.4f}")
print(f"- Accuracy:  {clustering_accuracy(filtered_ytrue, filtered_nmf_kmeans):.4f}")
print(f"- Silhouette:{silhouette_score(nmf_doc_topics, labels_nmf_kmeans):.4f}")
print(f"- CH:        {calinski_harabasz_score(nmf_doc_topics, labels_nmf_kmeans):.4f}")
print(f"- DB:        {davies_bouldin_score(nmf_doc_topics, labels_nmf_kmeans):.4f}")

# 3. Cluster with KMeans
nmf_spec = SpectralClustering(n_clusters=k_cluster)
labels_nmf_spec = nmf_kmeans.fit_predict(nmf_doc_topics)
#nmf_doc_topics = normalize(nmf_doc_topics, norm='l2')
# 4. Evaluate
filtered_nmf_spec = labels_nmf_spec[valid_idx]

print("\n📌 NMF → Spectral:")
print(f"- NMI:       {normalized_mutual_info_score(filtered_ytrue, filtered_nmf_spec):.4f}")
print(f"- Purity:    {purity_score_filtered(filtered_ytrue, filtered_nmf_spec, exclude_labels_from_majority=[-1]):.4f}")
print(f"- Accuracy:  {clustering_accuracy(filtered_ytrue, filtered_nmf_spec):.4f}")
print(f"- Silhouette:{silhouette_score(nmf_doc_topics, labels_nmf_spec):.4f}")
print(f"- CH:        {calinski_harabasz_score(nmf_doc_topics, labels_nmf_spec):.4f}")
print(f"- DB:        {davies_bouldin_score(nmf_doc_topics, labels_nmf_spec):.4f}")


###################################################################################
print("#########################################################################################")
print("\n CLUSTERIN  and TOPIC MODELING")
V_dense = V.toarray()

#HAC
############################################################################################

hac_model = AgglomerativeClustering(n_clusters=k_cluster)
final_labels_hac = hac_model.fit_predict(V_dense)  # V is TF-IDF
filtered_pred_labels_hac = final_labels_hac[valid_idx]

print("\n📌 HAC: ")

print(f"- NMI:       {normalized_mutual_info_score(filtered_ytrue, filtered_pred_labels_hac):.4f}")
print(f"- Purity:    {purity_score_filtered(filtered_ytrue, filtered_pred_labels_hac, exclude_labels_from_majority=[-1]):.4f}")
print(f"- Accuracy:  {clustering_accuracy(filtered_ytrue, filtered_pred_labels_hac):.4f}")
print(f"- Silhouette:{silhouette_score(V_dense, final_labels_hac):.4f}")
print(f"- CH:        {calinski_harabasz_score(V_dense, final_labels_hac):.4f}")
print(f"- DB:        {davies_bouldin_score(V_dense, final_labels_hac):.4f}")

###############################################################################################

#k-means
############################################################################################
kmeans_model = KMeans(n_clusters=k_cluster)
final_labels = kmeans_model.fit_predict(V_dense)
filtered_pred_labels = final_labels[valid_idx]

print("\n📌 k-means: ")

print(f"- NMI:       {normalized_mutual_info_score(filtered_ytrue, filtered_pred_labels):.4f}")
print(f"- Purity:    {purity_score_filtered(filtered_ytrue, filtered_pred_labels,exclude_labels_from_majority=[-1]):.4f}")
print(f"- Accuracy:  {clustering_accuracy(filtered_ytrue, filtered_pred_labels):.4f}")
print(f"- Silhouette:{silhouette_score(V_dense, final_labels):.4f}")
print(f"- CH:        {calinski_harabasz_score(V_dense, final_labels):.4f}")
print(f"- DB:        {davies_bouldin_score(V_dense, final_labels):.4f}")













i_nmh=np.where(ytrue_example==-1)[0]
i_mh= np.where(ytrue_example!= -1)[0]
def plot_all_tsne_models(models_info, true_labels, i_nmh, i_mh):
    """
    models_info: list of tuples (model_name, doc_topic_matrix, predicted_labels)
    """
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(14, 9))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    for idx, (model_name, topic_matrix, labels) in enumerate(models_info):
        row, col = divmod(idx * 2, 4)

        tsne = TSNE(n_components=2, perplexity=20, random_state=42)
        tsne_proj = tsne.fit_transform(topic_matrix)

        # Predicted cluster view
        axs[row, col].scatter(tsne_proj[:, 0], tsne_proj[:, 1], c=labels, cmap='tab20', s=6, alpha=0.8)
        axs[row, col].set_title(f"{model_name} (Pred.)")
        axs[row, col].axis('off')

        # Ground truth view
        axs[row, col+1].scatter(tsne_proj[i_nmh, 0], tsne_proj[i_nmh, 1], c=true_labels[i_nmh], cmap='tab20', s=6, alpha=0.6)
        axs[row, col+1].scatter(tsne_proj[i_mh, 0], tsne_proj[i_mh, 1], c=true_labels[i_mh], cmap='tab20', s=6, alpha=0.8)
        axs[row, col+1].set_title(f"{model_name} (GT)")
        axs[row, col+1].axis('off')

    fig.suptitle("t-SNE Comparison of Predicted Clusters and Ground Truth", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("tsne_grid_plot.pdf")  # For Springer submission
    plt.show()


models_info = [
    ("LDA→KMeans", lda_doc_topics, labels_lda_kmeans),
    ("LDA→HAC", lda_doc_topics, labels_lda_hac),
    ("NMF→KMeans", nmf_doc_topics, labels_nmf_kmeans),
    ("NMF→Spectral", nmf_doc_topics, labels_nmf_spec),
    ("HAC (TF-IDF)", V_dense, final_labels_hac),
    ("KMeans (TF-IDF)", V_dense, final_labels),
]

plot_all_tsne_models(models_info, ytrue_example, i_nmh, i_mh)



























# === Plotting: 2x2 layout ===
fig = plt.figure(figsize=(14, 10))

# Separated Model: Predicted Clusters
ax1 = fig.add_subplot(221)

scatter_sep = ax1.scatter(sep_tsne[:, 0], sep_tsne[:, 1], c=labels_sep, cmap='tab20', s=20, alpha=0.7)
ax1.set_title("CMTM Model - Predicted Clusters")
plt.colorbar(scatter_sep, ax=ax1, label="Predicted Cluster")
ax2 = fig.add_subplot(222)
i_nmh=np.where(ytrue_example==-1)[0]
i_mh= np.where(ytrue_example!= -1)[0]
#print(len(ytrue_example))

#print(i_mh)
#print(i_nmh)
#scatter_sep_gt = ax2.scatter(sep_tsne[:, 0], sep_tsne[:, 1], c=ytrue_example, cmap='tab20', s=20, alpha=0.7)
scatter_sep_gt = ax2.scatter(sep_tsne[i_nmh, 0], sep_tsne[i_nmh, 1], c=ytrue_example[i_nmh], cmap='tab20', s=20, alpha=0.7)
scatter_sep_gt = ax2.scatter(sep_tsne[i_mh, 0], sep_tsne[i_mh, 1], c=ytrue_example[i_mh], cmap='tab20', s=20, alpha=0.7)
ax2.set_title("CMTM Model - Ground Truth Labels")
plt.colorbar(scatter_sep_gt, ax=ax2, label="Ground Truth")


# Joint Model: Predicted Clusters
ax3 = fig.add_subplot(223)
scatter_int = ax3.scatter(int_tsne[:, 0], int_tsne[:, 1], c=labels_int, cmap='tab20', s=20, alpha=0.7)
ax3.set_title("COCNMF Model - Predicted Clusters")
plt.colorbar(scatter_int, ax=ax3, label="Predicted Cluster")
ax4 = fig.add_subplot(224)
#scatter_int_gt = ax4.scatter(int_tsne[:, 0], int_tsne[:, 1], c=ytrue_example, cmap='tab20', s=20, alpha=0.7)
scatter_int_gt = ax4.scatter(int_tsne[i_nmh, 0], int_tsne[i_nmh, 1], c=ytrue_example[i_nmh], cmap='tab20', s=20, alpha=0.7)
scatter_int_gt = ax4.scatter(int_tsne[i_mh, 0], int_tsne[i_mh, 1], c=ytrue_example[i_mh], cmap='tab20', s=20, alpha=0.7)
ax4.set_title("COCNMF Model - Ground Truth Labels")
plt.colorbar(scatter_int_gt, ax=ax4, label="Ground Truth")

plt.suptitle("t-SNE: CMTM vs COCNMF Models — Clusters vs Ground Truth", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()




"""TSNE plot for all the models"""
def plot_tsne_comparison(X_embed_input, predicted_labels, true_labels, title_prefix):
    tsne = TSNE(n_components=2, perplexity=20)
    tsne_proj = tsne.fit_transform(X_embed_input)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    scatter_pred = ax1.scatter(tsne_proj[:, 0], tsne_proj[:, 1], c=predicted_labels, cmap='tab20', s=20, alpha=0.7)
    ax1.set_title(f"{title_prefix} - Predicted Clusters")
    plt.colorbar(scatter_pred, ax=ax1, label="Cluster")

    scatter_gt = ax2.scatter(tsne_proj[i_nmh, 0], tsne_proj[i_nmh, 1], c=true_labels[i_nmh], cmap='tab20', s=20, alpha=0.7)
    scatter_gt = ax2.scatter(tsne_proj[i_mh, 0], tsne_proj[i_mh, 1], c=true_labels[i_mh], cmap='tab20', s=20, alpha=0.7)
    ax2.set_title(f"{title_prefix} - Ground Truth Labels")
    plt.colorbar(scatter_gt, ax=ax2, label="Ground Truth")

    plt.suptitle(f"t-SNE Visualization — {title_prefix}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
