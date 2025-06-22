import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import scipy.sparse as sp
import time
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import cProfile
from sklearn.metrics import silhouette_score
from sklearn.metrics import normalized_mutual_info_score
from collections import Counter, defaultdict
from ours import purity_score_filtered, train
start_time = time.time()

#The Algorithm
################################################################################
""" This code implements the update rules adapted from:

        Ebrahimi, S. F., & Peltonen, J. (2025). Constrained Non-negative Matrix Factorization for 
        Guided Topic Modeling of Minority Topics. 
        arXiv preprint arXiv:2505.16493."""
EPSILON= 1e-100

def _initialize_mmatrix(V, n_topics):
    m, n = V.shape
    W = np.abs(np.random.randn(m, n_topics) * 0.01)
    H = np.abs(np.random.randn(n_topics, n) * 0.01)
    return W, H

def kl_divergence(V, W, H):
    if sp.issparse(V):
        # compute np.dot(W, H) only where X is nonzero
        WH_data = _special_sparse_dot(W, H, V).data
        V_data = V.data
    else:
        WH = np.dot(W, H)
        WH_data = WH.ravel()
        V_data = V.ravel()

    indices = V_data > EPSILON
    WH_data = WH_data[indices]
    V_data = V_data[indices]
    # used to avoid division by zero

    WH_data[WH_data < EPSILON] = EPSILON

    V_data[V_data < EPSILON] = EPSILON

    sum_WH = np.dot(np.sum(W, axis=0), np.sum(H, axis=1))
    # computes np.sum(X * log(X / WH)) only where X is nonzero
    div = V_data / WH_data
    res = np.dot(V_data, np.log(div))
    # add full np.sum(np.dot(W, H)) - np.sum(X)
    res += sum_WH - V_data.sum()
    #return res

    num_documents = V.shape[0]
    num_vocab_terms = V.shape[1]
    return res / (num_documents * num_vocab_terms)




def safe_sparse_dot(a, b, *, dense_output=False):
    if a.ndim > 2 or b.ndim > 2:
        if sp.issparse(a):
            # sparse is always 2D. Implies b is 3D+
            # [i, j] @ [k, ..., l, m, n] -> [i, k, ..., l, n]
            b_ = np.rollaxis(b, -2)
            b_2d = b_.reshape((b.shape[-2], -1))
            ret = a @ b_2d
            ret = ret.reshape(a.shape[0], *b_.shape[1:])
        elif sp.issparse(b):
            # sparse is always 2D. Implies a is 3D+
            # [k, ..., l, m] @ [i, j] -> [k, ..., l, j]
            a_2d = a.reshape(-1, a.shape[-1])
            ret = a_2d @ b
            ret = ret.reshape(*a.shape[:-1], b.shape[1])
        else:
            ret = np.dot(a, b)
    else:
        ret = a @ b

    if (
            sp.issparse(a)
            and sp.issparse(b)
            and dense_output
            and hasattr(ret, "toarray")
):
        return ret.toarray()
    return ret


def _special_sparse_dot(W, H, X):
    """Computes np.dot(W, H), only where X is non zero."""
    if sp.issparse(X):
        ii, jj = X.nonzero()
        n_vals = ii.shape[0]
        dot_vals = np.empty(n_vals)
        n_components = W.shape[1]

        batch_size = max(n_components, n_vals // n_components)
        for start in range(0, n_vals, batch_size):
            batch = slice(start, start + batch_size)
            dot_vals[batch] = np.multiply(W[ii[batch], :], H.T[jj[batch], :]).sum(axis=1)
        WH = sp.coo_matrix((dot_vals, (ii, jj)), shape=X.shape)
        return WH.tocsr()
    else:
        return np.dot(W, H)

def _special_sparse_div(V, WH):
    """Computes np.dot(W, H), only where X is non zero."""
    if sp.issparse(V):
        ii, jj = V.nonzero()
        n_vals = ii.shape[0]
        dot_vals = V[(ii,jj)]/ WH[(ii, jj)]
        VWH = sp.coo_matrix((np.array(dot_vals).squeeze(), (ii, jj)), shape=V.shape)
        return VWH.tocsr()
    else:
        return V/WH


def g1(W, MH_indices, I_0, W_max):
    """
    Constraint function g1 to penalize MH topic weights in non-MH docs.
    """
    g1_matrix = np.zeros_like(W)
    for i in I_0:
        for k in MH_indices:
            g1_matrix[i, k] = W[i, k] - W_max
    return g1_matrix

def update_lambda(lambda_, W, MH_indices, I_0, W_max, eta):
    g1_val = g1(W, MH_indices, I_0, W_max)
    lambda_ = np.maximum(0, lambda_ + eta * g1_val)
    lambda_[g1_val < 0] = 0
    return lambda_


def g2(H, seed_indices, theta_min):
    num = np.sum(H[:, seed_indices], axis=1)
    den = np.sum(H, axis=1)
    g2_value = theta_min - (num / den)
    return g2_value


def update_mu(mu, H, seed_indices, theta_min, eta):
    g2_val = g2(H, seed_indices, theta_min)
    g2_val_expanded = g2_val[:, np.newaxis]
    mu_update = mu + eta * g2_val_expanded
    mu = np.maximum(0, mu_update)
    mu[g2_val < 0] = 0
    return mu

def update_H(V, W, H, WH, mu, seed_indices, MH_indices):
    #WH =_special_sparse_dot(W, H, V)
    V_WH=_special_sparse_div(V, WH)
    positive_term = safe_sparse_dot(W.T, V_WH)
    negative_term = np.sum(W, axis=0)[:, np.newaxis]  # shape (k, 1)
    num = np.sum(H[:, seed_indices], axis=1, keepdims=True)
    den = np.sum(H, axis=1, keepdims=True)
    den = np.maximum(den, EPSILON)
    g2_term = np.zeros_like(H)
    for k in range(H.shape[0]):
        if k in MH_indices:
            g2_term[k, :] = num[k] / (den[k] ** 2+ EPSILON)
            g2_term[k, seed_indices] = -((den[k] - num[k]) / (den[k] ** 2+ EPSILON))
    H *= positive_term / (negative_term + mu* g2_term)
    return H

def update_W2(V, W, H, lambda_, MH_indices, I_0):
    WH = _special_sparse_dot(W, H, V)
    V_div_WH = _special_sparse_div(V, WH)

    positive_term = safe_sparse_dot(V_div_WH, H.T)  # shape (n, k)
    negative_term = np.sum(H, axis=1)[np.newaxis, :]  # shape (1, k)

    g1_W = np.zeros_like(W)
    for i in I_0:
        for k in MH_indices:
            g1_W[i, k] = lambda_[i, k]  # actual penalty value from λ

    W *= positive_term / (negative_term + g1_W)
    return W

#####################################################################################
"""Our contributions are as follows updates"""

def update_b_tilde(V, H, WH, b_tilde, p, a_tilde, e_tilde, MH_indices, lambda3, I_0, eta_b):
    """Update b_tilde using reparameterized gradient."""
    p_a = np.dot(p, a_tilde[:, MH_indices] ** 2) + e_tilde[:, MH_indices] ** 2
    b2 = b_tilde ** 2
    H_MH = H[MH_indices, :]
    diff = 1 - V.toarray() / (WH + EPSILON)
    grad_b = np.einsum('nm,km,nk->n', diff, H_MH, p_a) * 2 * b_tilde
    mask = np.isin(np.arange(b_tilde.shape[0]), I_0).astype(float)
    grad_b += 4 * lambda3 * (b_tilde ** 3) * mask
    b_tilde -= eta_b * grad_b
    b_tilde = np.clip(b_tilde, EPSILON, 10)
    return b_tilde

def update_a_tilde(V, H, WH, b_tilde, p, a_tilde, e_tilde, MH_indices, lambda4, eta_a):
    grad_a = np.zeros_like(a_tilde)
    b2 = b_tilde ** 2
    for k_idx in MH_indices:
        Hk = H[k_idx, :]
        diff = 1 - V.toarray() / (WH + EPSILON)
        residual = diff * Hk[np.newaxis, :]
        weighted_p = b2[:, np.newaxis] * p
        grad_ck = np.einsum('nm,nc->cm', residual, weighted_p)
        grad_a[:, k_idx] = 2 * a_tilde[:, k_idx] * grad_ck.sum(axis=1) + 4 * lambda4 * a_tilde[:, k_idx] ** 3
    a_tilde -= eta_a * grad_a
    a_tilde== np.clip(a_tilde, EPSILON, 10)
    return a_tilde

def update_e_tilde(V, H, WH, b_tilde, p, a_tilde, e_tilde, MH_indices, lambda2, eta_e):
    diff = 1 - V.toarray() / (WH + EPSILON)
    b2 = b_tilde ** 2
    grad_e = np.zeros_like(e_tilde)
    for k_idx in MH_indices:
        Hk = H[k_idx, :]
        residual = diff * Hk[np.newaxis, :]
        grad_KL = np.sum(residual, axis=1) * b2
        grad_e[:, k_idx] = 2 * e_tilde[:, k_idx] * grad_KL + 4 * lambda2 * e_tilde[:, k_idx] ** 3
    e_tilde -= eta_e * grad_e
    e_tilde = np.clip(e_tilde, EPSILON, 10)
    return e_tilde



def softmax(z, tau=0.5):
    z = z / tau
    z -= np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def update_pic_optimized(V, H, WH, b, z, a, e, MH_indices, lambda1, p, eta_p, lambda_H, sparsity_p):
    """Gradient descent update for z (used to compute p), with sparsity and entropy regularization."""
    n, C = z.shape
    grad_z = np.zeros_like(z)
    diff = 1 - V.toarray() / (WH + EPSILON)  # (n, m)

    # --- Reconstruction-based gradients ---
    for k_idx in MH_indices:
        Hk = H[k_idx, :]  # (m,)
        weighted_diff = diff * Hk[np.newaxis, :]  # (n, m)
        sum_pa = np.dot(p, a[:, k_idx][:, np.newaxis])  # (n, 1)

        for c in range(C):
            delta = (a[c, k_idx] - sum_pa[:, 0]) * b  # (n,)
            grad_core = np.sum(weighted_diff * delta[:, np.newaxis], axis=1)  # (n,)
            grad_z[:, c] += grad_core  # Removed * p[:, c] for sharper update


    # --- Sparsity ℓ_p regularization (vectorized) ---
    grad_sparse = lambda1 * sparsity_p * (p + EPSILON) ** (sparsity_p - 1)
    grad_z += grad_sparse
    # Without entropy, the optimization might get stuck with one or two dominant clusters.
    # --- Entropy regularization ---
    entropy_grad = lambda_H * (-p * (1 + np.log(p + EPSILON)))
    grad_z += entropy_grad
    grad_z = np.clip(grad_z, EPSILON, 10)

    # --- Gradient update step ---
    z -= eta_p * grad_z
    return z


def compute_W1(tilde_b, p, tilde_a, tilde_e, MH_indices):
    n, k = tilde_e.shape
    W1 = np.zeros((n, k))
    b_sq = tilde_b**2
    for i in range(n):
        for k_idx in MH_indices:
            W1[i, k_idx] = b_sq[i] * (np.sum(p[i, :] * (tilde_a[:, k_idx]**2)) + (tilde_e[i, k_idx]**2))
    return W1

def update_lambda_W1(lambda_W1, W1, MH_indices, I_0, epsilon, eta_lambda):
    for i in I_0:
        for k in MH_indices:
            violation = W1[i, k] - epsilon
            lambda_W1[i, k] = max(0, lambda_W1[i, k] + eta_lambda * violation)
    return lambda_W1

#######################################################################################

def initialize_model_reparam(V, k, C, seed_indices):
    """
    Initialize reparameterized model parameters and dual variables.
    Returns tilde versions of b, a, e.
    """
    n, m = V.shape

    # === W and H ===
    W = np.random.rand(n, k) * 0.05
    H = np.random.rand(k, m) * 0.05

    # === b_tilde: higher for seed docs ===
    b_tilde = 0.05 * np.ones(n)
    doc_seedword_sums = np.array(V[:, seed_indices].sum(axis=1)).flatten()
    seed_rich_doc_indices = np.where(doc_seedword_sums > 0)[0]
    b_tilde[seed_rich_doc_indices] = np.sqrt(0.3)  # because b_i = b_tilde^2

    # === z and p ===
    z = np.random.normal(scale=0.1, size=(n, C))
    p = np.exp(z)
    p /= np.sum(p, axis=1, keepdims=True)

    # === a_tilde ===
    a_tilde = np.random.rand(C, k) * 0.1

    # === e_tilde ===
    e_tilde = np.random.randn(n, k) * 0.1

    # === duals ===
    lambda_ = np.zeros((n, k))
    mu = np.zeros((k, m))
    lambda_W1 = np.zeros((n, k))

    return W, H, b_tilde, z, p, a_tilde, e_tilde, lambda_, mu, lambda_W1






def full_training_loop(V, n_iter, lambda1, lambda2, lambda3, lambda4,
                       eta_b, eta_a, eta_e, eta_p, k, C, 
                       MH_indices, I_0, seed_indices, 
                       lambda_H, sparsity_p):
    """
    Full alternating minimization loop for the model.
    """
    # Tracking loss
    n, m = V.shape

    losses = []
    W, H, tilde_b, z, p, tilde_a, tilde_e, lambda_, mu, lambda_W1 = initialize_model_reparam(V, k, C, seed_indices)

    W1 = np.zeros_like(W)
    
    for it in range(n_iter):
        W1 = compute_W1(tilde_b, p, tilde_a, tilde_e, MH_indices)
        W_combined = W.copy()
        W_combined[:, MH_indices] = W1[:, MH_indices]

        WH = _special_sparse_dot(W_combined, H, V).toarray()

        # === H Update ===
        H = update_H(V, W_combined, H, WH, mu, seed_indices, MH_indices)

        # === W2 Update ===
        W = update_W2(V, W, H, lambda_, MH_indices, I_0)

        # === lambda and mu updates ===
        lambda_W1 = update_lambda_W1(lambda_W1, W1, MH_indices, I_0, epsilon=1e-6, eta_lambda=0.001)
        lambda_ = update_lambda(lambda_, W_combined, MH_indices, I_0, W_max=1e-9, eta=0.001)
        mu = update_mu(mu, H, seed_indices, theta_min=0.4, eta=0.001)
        
        # === Remaining parameter updates ===
        #p_norm = np.power(np.abs(p), 0.3)
        #lambda_W1 = update_lambda_W1(lambda_W1, W1, MH_indices, I_0, epsilon=EPSILON, eta_lambda=0.01)

        # === Step 6: Gradient descent updates for structured variables ===
        tilde_b = update_b_tilde(V, H, WH, tilde_b, p, tilde_a, tilde_e, MH_indices, lambda3, I_0, eta_b)
        tilde_a = update_a_tilde(V, H, WH, tilde_b, p, tilde_a, tilde_e,MH_indices, lambda4, eta_a)
        tilde_e = update_e_tilde(V, H, WH, tilde_b, p, tilde_a, tilde_e, MH_indices, lambda2, eta_e)

        # === Step 7: Cluster membership softmax logits update (reparameterized p_ic) ===
        z = update_pic_optimized(V, H, WH, tilde_b**2, z, tilde_a**2, tilde_e**2,
                                 MH_indices, lambda1, p, eta_p, lambda_H, sparsity_p)
        p = softmax(z, tau=0.3)
        #p /= np.maximum(p.sum(axis=1, keepdims=True), EPSILON)
        
        # === Step 8: Loss tracking ===
        loss = kl_divergence(V, W_combined, H)
        losses.append(loss)
        if it % 2 == 0 or it == n_iter - 1:
            print(f"Iteration {it+1}/{n_iter}, KL Loss: {loss:.5f}")

    return W_combined, H, tilde_b, tilde_a, tilde_e, p, losses
