from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
import re, nltk, os
import numpy as np

# For Part-Of_speech (POS)
nltk.download("averaged_perceptron_tagger_eng")


def lemmatize(tokens):
    """
    Lemmatized the tokens which means to convert the tokens to
    their most basic canonical form.

    Lemmatization examples:
    running -> run, walked -> walk

    :return: a string of all the words lemmatize, can be used to write to corpus.
    """
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


def tokenize(text):
    """
    Remove uppercase and special chars of a text (str)
    And return an array of tokens (in our case it's just words)
    """

    text = str(text).lower()
    text = re.sub(r"[^ a-z]", "", text)

    stop_words = nltk.corpus.stopwords.words("english")

    return [word for word in text.split() if word not in stop_words and len(word) <= 25]


def tokenize_and_lemmatize(text):
    tokens = tokenize(text)
    res = lemmatize(tokens)
    return res


def search_document(n, query, corpus):
    print("Finding relevant documents...")

    # Initialize the vectorizer that calculate all the TFIDF
    vectorizer = TfidfVectorizer()

    query_normalized = tokenize_and_lemmatize(query)

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


def save_corpus(corpus, file_path):
    with open(file_path, mode="w", encoding="utf8") as fp:
        for document in corpus:
            fp.write("%s\n" % document)


def create_corpus(read_directory, write_filename):
    corpus = []

    with os.scandir(read_directory) as file_paths:

        for file_path in file_paths:
            if not file_path.is_file():
                continue

            with open(file_path, mode="r", encoding="utf8") as file:
                content = file.read()
                res = tokenize_and_lemmatize(content)
                corpus.append(res)

    save_corpus(corpus, write_filename)

    print("Corpus saved to: " + write_filename)


create_corpus("./documents/", "./corpus.txt")
