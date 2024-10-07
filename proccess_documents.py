import os
import shutil

from functions import tokenize_and_lemmatize


def save_corpus(corpus, file_path):
    with open(file_path, mode="w", encoding="utf8") as fp:
        for document in corpus:
            fp.write("%s\n" % document)


def create_corpus(read_directory, write_filename, remove_originals=False):
    corpus = []
    formatted_directory = "./formatted_documents/"

    # Remove everything in the directory if it already exists
    if os.path.exists(formatted_directory):
        shutil.rmtree(formatted_directory)

    # Create a new directory for renamed files
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


create_corpus("./documents", "corpus.txt")
