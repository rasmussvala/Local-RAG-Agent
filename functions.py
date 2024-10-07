import re
import nltk


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
