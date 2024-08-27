from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
import re, nltk
import numpy as np

# For Part-Of_speech (POS)
nltk.download("averaged_perceptron_tagger_eng")


def lemmatize(text):
    # Remove uppercase and special chars
    text = text.lower()
    text = re.sub(r"[^ a-z]", "", text)

    stop_words = nltk.corpus.stopwords.words("english")

    tokens = [
        word for word in text.split() if word not in stop_words and len(word) <= 25
    ]

    lemmatizer = nltk.wordnet.WordNetLemmatizer()

    tagged_tokens = nltk.pos_tag(tokens, tagset=None)

    lemmatized_words = []

    for token, tag in tagged_tokens:
        if tag.startswith("V"):  # Verb
            pos_val = "v"

        elif tag.startswith("J"):  # Adjective
            pos_val = "a"

        elif tag.startswith("R"):  # Adverb
            pos_val = "r"

        else:
            pos_val = "n"  # Noun

        lemmatized_word = lemmatizer.lemmatize(token, pos_val)

        lemmatized_words.append(lemmatized_word)

    return " ".join(lemmatized_words)


def lemmatize_and_remove_stopwords(text):
    print("hi")


def search_document(n, query, corpus):
    print("Finding relevant documents...")

    # Initialize the vectorizer that calculate all the TFIDF
    vectorizer = TfidfVectorizer()

    query_normalized = lemmatize_and_remove_stopwords(query)

    # Create a matrix full of vectors with calculated TFIDFs
    corpus_TFIDF = vectorizer.fit_transform(corpus)
    # Fit query in the matrix to be able to calculate the similarities
    query_TFIDF = vectorizer.transform([query_normalized])

    # calculate the cosine distances and flatten the array since query is one dimensional
    similarities = cosine_distances(corpus_TFIDF, query_TFIDF)
    similarities = np.ravel(similarities)

    # Find the n most similar indices and format them like 0001, 0002 etc.
    n_most_similar_indices = np.argsort(similarities)[:n]
    n_most_similar_indices = [str(i).zfill(4) for i in n_most_similar_indices]


lemmatize("hello my name is rasmus and i want to be a race car driver")
