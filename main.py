import shutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
import re
import nltk
import os
import numpy as np
from assistant import start_chat_session

# Download necessary NLTK data
nltk.download("averaged_perceptron_tagger_eng")  # For Part-Of_speech (POS)
nltk.download("stopwords")
nltk.download("wordnet")


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
    print("\nFinding relevant documents...")

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
    n_most_similar_distances = similarities[n_most_similar_indices]
    n_most_similar_indices = [str(i).zfill(4) for i in n_most_similar_indices]

    return n_most_similar_indices, n_most_similar_distances


def save_corpus(corpus, file_path):
    with open(file_path, mode="w", encoding="utf8") as fp:
        for document in corpus:
            fp.write("%s\n" % document)


def create_corpus(read_directory, write_filename, remove_originals=False):
    corpus = []
    formatted_directory = "./formatted_documents/"

    # Create a new directory for renamed files if it doesn't exist
    if not os.path.exists(formatted_directory):
        os.makedirs(formatted_directory)

    with os.scandir(read_directory) as file_paths:
        for index, file_path in enumerate(file_paths):
            if not file_path.is_file():
                continue

            # Generate new filename with numeric prefix
            new_filename = f"{index:04d}_{file_path.name}"
            new_file_path = os.path.join(formatted_directory, new_filename)

            # Copy and rename the file
            shutil.copy2(file_path.path, new_file_path)

            with open(new_file_path, mode="r", encoding="utf8") as file:
                content = file.read()
                res = tokenize_and_lemmatize(content)
                corpus.append(res)

            # Remove the original file if specified
            if remove_originals:
                os.remove(file_path.path)

    save_corpus(corpus, write_filename)

    print(f"Files renamed and copied to: {formatted_directory}")
    if remove_originals:
        print(f"Original files in {read_directory} have been removed.")
    print(f"Corpus saved to: {write_filename}")


def read_corpus(file_path):

    corpus = []

    with open(file_path, mode="r", encoding="utf8") as file:
        for line in file:
            corpus.append(line)

    return corpus


def retrive_relevant_content(similar_indices, path_to_documents):
    relevant_content = []

    for index in similar_indices:
        for filename in os.listdir(path_to_documents):
            if not filename.startswith(str(index)):
                continue

            file_path = path_to_documents + filename
            with open(file_path, mode="r", encoding="utf8") as file:
                content = ""

                for line in file:
                    content += line

                relevant_content.append(content)

    return relevant_content


def initialize_assistant():
    path_to_documents = "./documents/"
    path_to_formatted_documents = "./formatted_documents/"
    path_to_corpus = "./corpus.txt"

    # create_corpus(path_to_documents, path_to_corpus)

    corpus = read_corpus(path_to_corpus)

    welcome_message = (
        "\n********************************************************************\n"
        + "Welcome to a local RAG Agent with access to your internal documents.\n"
        + 'To end the conversation, say "Goodbye"\n'
        + "********************************************************************\n"
    )

    print(welcome_message)

    print("You: ", end="")
    query = input()

    n = 3
    similar_indices, distances = search_document(n, query, corpus)

    print_found_documents(n, similar_indices, distances)

    relevant_content = retrive_relevant_content(
        similar_indices, path_to_formatted_documents
    )

    start_chat_session(query, relevant_content)


def print_found_documents(n, similar_indices, distances):
    for i in range(n):
        print(
            f"Found document with index: {similar_indices[i]} and cosine distance: {distances[i]:.2f}"
        )


if __name__ == "__main__":
    initialize_assistant()
