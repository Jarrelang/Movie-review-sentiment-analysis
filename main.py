import pandas as pd
import os
import contractions
from bs4 import BeautifulSoup
import spacy 
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


# functions
def remove_html(text):
    return BeautifulSoup(text, "html.parser").get_text()

def expand_contractions(text):
    return contractions.fix(text)

def preprocess_spacy(texts):
    cleaned_reviews = []

    for doc in tqdm(nlp.pipe(texts, batch_size=1000, disable=["parser", "ner"], n_process=os.cpu_count() // 2), total=len(texts)):
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and not token.is_space
        ]
        cleaned_reviews.append(" ".join(tokens))

    return cleaned_reviews

def spacyfy_single_review(text):
    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
    ]
    return " ".join(tokens)

# load spacy pipeline
nlp = spacy.load('en_core_web_sm')


if __name__ == "__main__":

    # if data already cleaned, just use it
    if os.path.exists("IMDB_cleaned.csv"):
        print("Found processed data! Loading from 'IMDB_cleaned.csv'...")
        dataset = pd.read_csv("IMDB_cleaned.csv")

        # get rid of null/empty data
        dataset.dropna(inplace=True)

    # data has not been cleaned so perform the cleaning
    else:

        # load dataset
        dataset = pd.read_csv("IMDBdataset.csv")

        # remove NaNs
        dataset.dropna(inplace=True)

        # remove duplicate data
        dataset = dataset.drop_duplicates()

        # turn sentiments from str to int
        dataset['sentiment'] = dataset['sentiment'].replace({
            'positive': 1,
            'negative': 0
        })

        # remove html tags
        dataset['review'] = dataset['review'].apply(remove_html)

        # expand words like didn't, isn't, you're etc.
        dataset['review'] = dataset['review'].apply(expand_contractions)

        # lower case all reviews
        dataset['review'] = dataset['review'].str.lower()

        # tokenise, remove stop words & punctuations & lemmatise
        dataset['review'] = preprocess_spacy(dataset['review'])

        # save cleaned data
        print("Saving processed data to 'IMDB_cleaned.csv'...")
        dataset.to_csv("IMDB_cleaned.csv", index=False)

    # removes rows that are empty or whitespace only
    dataset = dataset[dataset['review'].str.strip().astype(bool)]

    # split dataset for training and testing
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    dataset['review'], 
    dataset['sentiment'], 
    test_size=0.2, 
    random_state=42,
    stratify=dataset['sentiment']
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1, 3))),
        ('chi2', SelectKBest(score_func=chi2, k=5000)),
        ('clf', LinearSVC(dual='auto', random_state=42, C=1))
    ])

    # average model performance using training set
    cv_scores = cross_val_score(pipeline, X_train_raw, y_train, cv=5)
    print(f"Accuracy calculated using training set: {cv_scores.mean()*100:.2f}%")

    # train model
    print("Training final model on full training set...")
    pipeline.fit(X_train_raw, y_train)

    # predict labels for X_test and calculate accuracy
    test_accuracy = pipeline.score(X_test_raw, y_test)
    print(f" -> Final Test Accuracy: {test_accuracy*100:.2f}%")
    print("-" * 30)

    while True:
        # get user input to analyse a specific review
        new_review = input("Input a review here: ")

        # enter q to quit
        if new_review == "q":
            break

        # clean user input
        cleaned_new_review = remove_html(new_review)
        cleaned_new_review = expand_contractions(cleaned_new_review)
        cleaned_new_review = cleaned_new_review.lower()
        cleaned_new_review = spacyfy_single_review(cleaned_new_review)

        # if cleaned input is not an empty string, vectorize and choose features
        if cleaned_new_review.strip():
            prediction = pipeline.predict([cleaned_new_review])

            if prediction[0] == 1: # prediction returns a list like [1], so we get index 0
                print("I think it is a positive review!")
            else:
                print("I think it is a negative review!")

        else:
            print("The review was empty or contains only stop words")

