import shutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
import re
import nltk
import os
import numpy as np
from assistant import start_chat_session
from functions import tokenize_and_lemmatize

# Download necessary NLTK data
nltk.download("averaged_perceptron_tagger_eng")  # For Part-Of_speech (POS)
nltk.download("stopwords")
nltk.download("wordnet")


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
