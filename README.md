🧠 COCNMF: Cluster-Oriented Constrained Non-negative Matrix Factorization

COCNMF is a joint cluster-topic modeling algorithm designed to discover minority topics, such as mental health themes, utilizing seed word constraints and a joint clustering structure.

This repository implements the algorithm proposed in our paper and provides a reproducible pipeline for evaluating performance on synthetic datasets (e.g., blended Reuters + 20 Newsgroups and mental health social media documents).



---

### ⚠️ Review-Only Notice

This repository is provided **solely for peer review purposes**.

- The code is **not licensed for commercial or public use**.
- Redistribution, reuse, or publication of the code or data is **not permitted**.
- For questions, contact the author at [seyedeh.ebrahimi@tuni.fi](mailto:seyedeh.ebrahimi@tuni.fi).

If you're a reviewer, thank you for evaluating this work!

---

```plaintext
| File                                         | Description                                   |
| -------------------------------------------- | --------------------------------------------- |
| requirements.txt`                            | dependecies and packages                      |
| `run.py`                                     | Main script to train the COCNMF model         |
| `eval.py`                                    | The main evaluation script                    |
| `cmtm_Algorithm.py`                          | The main algorithm adopted for evaluation     |
| `t_test_run.py`                              | T-test based comparison of clustering results |
| `param_tuning.py`                            | Hyperparameter tuning script                  |
| `mixed_reuters_20ng_synthetic_style_org.csv` | Synthetic benchmark dataset                   |
| `stopwords-fi.txt`                           | Finnish stopwords used in preprocessing       |
| `.png / .pdf files`                          | Evaluation and visualization outputs          |
```




# 🚀 How to Run
```bash
git clone https://github.com/seyedeh-mona-ebrahimi/COCNMF.git
cd COCNMF
```

## (Recommended) create and activate a fresh virtual environment first
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .\.venv\Scripts\activate   # Windows PowerShell
```

### Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```




## 🚀 Run the Main Pipeline

> To **train the model** and evaluate it:

```bash
python run.py
```

> To **evaluate the model across various baselines** :

```bash
python eval.py
```


> To **tune parameters** over a grid:
```bash
python param_tuning.py
```


> To run t-tests comparing different model variants:

```bash
python t_test_run.py
```
