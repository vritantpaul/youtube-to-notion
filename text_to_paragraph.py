import math
import spacy
import numpy as np
from scipy.signal import argrelextrema
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def consistent_sentences(sentences):
    """
    Makes sentences consistent in size by concatenating shorter sentences
    and splitting longer sentences.
    """
    # GET THE LENGTH OF EACH SENTENCE ---
    length = [len(sentence) for sentence in sentences]

    # DETERMINE LONGEST OUTLIER ---
    long = np.mean(length) + np.std(length) * 2
    # DETERMINE SHORTEST OUTLIER ---
    short = np.mean(length) - np.std(length) * 1.25

    # SPLITTING LONG SENTENCES AND CONCATENATING SHORT ONES ---
    text = ""
    for sentence in sentences:
        if len(sentence) > long and "," in sentence and '"' not in sentence:
            line = sentence.replace(", ", ".\n", 1)
            text += line + "\n"
        elif len(sentence) < short:
            text += sentence + " "
        else:
            text += sentence + "\n"

    # CAPITALIZING FIRST LETTER OF SPLIT SENTENCES ---
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if (letter := line[0]).islower():
            lines[i] = line.replace(letter, letter.capitalize(), 1)

    return lines


def activate_similarities(similarities: np.array, p_size=10) -> np.array:
    # sourcery skip: inline-immediately-returned-variable
    """
    Function returns list of weighted sums of activated sentence similarities
    Args:
        similarities (numpy array): it should square matrix where each sentence corresponds to another with cosine similarity
        p_size (int): number of sentences are used to calculate weighted sum
    Returns:
        list: list of weighted sums
    """
    # REVERSE SIGMOID FUNCTION ---
    rev_sigmoid = lambda x: (1 / (1 + math.exp(0.5 * x)))

    # CREATING SPACE TO CREATE WEIGHTS FOR SIGMOID FUNCTION ---
    x = np.linspace(
        -10, 10, p_size
    )  # "p_size" will determine number of sentences used and the size of weights vector.

    # APPLYING ACTIVATION FUNCTION TO THE CREATED SPACE ---
    y = np.vectorize(rev_sigmoid)

    # ADDING ZEROES TO NEGLECT THE EFFECT OF EVERY ADDITIONAL SENTENCE & MULTIPLYING TO MATCH THE LENGTH OF THE VECTOR ---
    # (as we are only applying activation to p_size number of sentences)
    activation_weights = np.pad(y(x), (0, similarities.shape[0] - p_size))

    # Step 1: TAKE EACH DIAGONAL TO THE RIGHT OF THE MAIN DIAGONAL
    diagonals = [similarities.diagonal(each) for each in range(similarities.shape[0])]
    # Step 2: PAD EACH DIAGONAL BY ZEROS AT THE END, BECAUSE EACH DIAGONAL IS OF DIFFERENT LENGTH WE SHOULD PAD IT WITH ZEROS AT THE END
    diagonals = [
        np.pad(each, (0, similarities.shape[0] - len(each))) for each in diagonals
    ]
    # Step 3: STACK THOSE DIAGONALS INTO NEW MATRIX
    diagonals = np.stack(diagonals)
    # Step 4: APPLY ACTIVATION WEIGHTS TO EACH ROW. MULTIPLY SIMILARITIES WITH OUR ACTIVATION.
    diagonals = diagonals * activation_weights.reshape(-1, 1)
    # Step 5: CALCULATE THE WEIGHTED SUM OF ACTIVATED SIMILARITIES
    activated_similarities = np.sum(diagonals, axis=0)
    return activated_similarities


def split_to_paragraphs(text):
    """
    Putting it all together:
    - Splitting text into sentences
    - Making sentences consistent in length
    - Dividing the text into paragraphs based on sentence similarity.
    """
    # LOADING LANGUAGE MODELS ---
    nlp = spacy.load("en_core_web_sm")
    model = SentenceTransformer("all-mpnet-base-v2")

    # SPLITTING TEXT TO LINES ---
    doc = nlp(text)
    lines = [sent.text for sent in doc.sents]

    # MAKING SENTENCE LENGTH CONSISTENT ---
    sentences = consistent_sentences(lines)

    # EMBED(VECTORIZE) SENTENCE ---
    embeddings = model.encode(sentences)

    # CREATING A SIMILARITY MATRIX ---
    similarities = cosine_similarity(embeddings)

    # IDENTIFYING SPLITS POINTS ---
    activated_similarities = activate_similarities(similarities, p_size=10)
    # FOR ALL LOCAL MINIMAS FINDING RELATIVE MINIMA OF OUR VECTOR & SAVE THEM TO VARIABLE WITH ARGRELEXTREMA FUNCTION
    minmimas = argrelextrema(activated_similarities, np.less, order=4)
    # NOTE: order parameter controls how frequent should be splits. Higher the order lower the number of splits

    # GET THE ORDER NUMBER OF THE SENTENCES WHICH ARE IN SPLITTING POINTS
    split_points = list(minmimas[0])

    # FINALLY CREATING THE TEXT ---
    text = "".join(
        "\n\n" + sentence + " " if index in split_points else sentence + " "
        for index, sentence in enumerate(sentences)
    )
    return text
