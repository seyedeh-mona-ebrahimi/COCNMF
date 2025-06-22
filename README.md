🧠 COCNMF: Cluster-Oriented Constrained Non-negative Matrix Factorization

COCNMF is a semi-supervised topic modeling and document clustering algorithm designed for discovering minority topics, such as mental health themes, using seed word constraints and joint clustering structure.

This repository implements the algorithm proposed in our paper and provides a reproducible pipeline for evaluating performance on synthetic datasets (e.g., blended Reuters + 20 Newsgroups).

| File                                         | Description                                   |
| -------------------------------------------- | --------------------------------------------- |
| `run.py`                                     | Main script to train the COCNMF model         |
| `eval.py`                                    | The main evaluation script                    |
| `t_test_run.py`                              | T-test based comparison of clustering results |
| `param_tuning.py`                            | Hyperparameter tuning script                  |
| `mixed_reuters_20ng_synthetic_style_org.csv` | Synthetic benchmark dataset                   |
| `stopwords-fi.txt`                           | Finnish stopwords used in preprocessing       |
| `.png / .pdf files`                          | Evaluation and visualization outputs          |


git clone https://github.com/seyedeh-mona-ebrahimi/COCNMF.git
cd COCNMF


pip install -r requirements.txt
