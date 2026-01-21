# Movie Review Sentiment Analyser

This project is a sentiment analyser for movie reviews. It trains a Linear SVM classifier on the IMDB movie review dataset to predict whether a given review is positive or negative.

---

# Features

* HTML tag removal and contraction expansion
* Tokenisation, stop-word removal, and lemmatisation using **spaCy**
* TF-IDF vectorisation with **uni-, bi-, and tri-grams**
* Feature selection using **Chi-squared**
* Classification using **Linear Support Vector Machine (LinearSVC)**
* 5-fold cross-validation on training data
* Interactive CLI for real-time sentiment prediction

---

> **Note:** The full IMDB dataset (IMDBdataset.csv) and cleaned dataset 
(IMDB_cleaned.csv) are intentionally excluded due to size.

---

# Dataset

This project uses the IMDB Movie Reviews Dataset, containing labelled reviews:

* review: movie review text
* sentiment: positive or negative

# Required files (not committed to GitHub)

IMDB Dataset of 50K Movie Reviews has not been included due to size 
and is required for the code to run

Download the file here https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data
and name it IMDBdataset.csv

---

# Install dependencies

```bash
pip install -r requirements.txt
```

# Download spaCy language model

```bash
python -m spacy download en_core_web_sm
```

---

### How It Works ###

Each review undergoes the following steps:

1. HTML tag removal (`BeautifulSoup`)
2. Contraction expansion (e.g. *"didn't" → "did not"*)
3. Lowercasing
4. spaCy-based:

   * Tokenisation
   * Stop-word removal (words that do not provide much meaning such as 'the', 'is' etc)
   * Punctuation removal
   * Lemmatization (converting every word into its base form ie. running -> run)

# Machine Learning Pipeline

Implemented using `sklearn.pipeline.Pipeline`:

1. # TF-IDF Vectorizer

   * Max features: 20,000
   * N-grams: (1, 3)

2. # Chi-Squared Feature Selection

   * Top 5,000 features

3. # Linear SVM (LinearSVC)

---

# Training & Evaluation

* Dataset split: 80% train / 20% test
* Stratified split to preserve sentiment balance
* 5-fold cross-validation on training data
* Final evaluation on held-out test set

---

### Interactive Prediction Mode

After training, you can enter your own reviews to test the model

---