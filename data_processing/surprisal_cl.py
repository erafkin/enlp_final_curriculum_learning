from collections import defaultdict, Counter
import re
from typing import Dict
import numpy as np


def tokenize(text: str):
    # Basic tokenizer: lowercase and split on non-word characters
    return re.findall(r'\b\w+\b', text.lower())


def generate_trigram_probabilities(corpus) -> Dict:
    """
        get a trigram probability table of the corpus
    """
    if isinstance(corpus, str):
        sentences = [corpus]
    else:
        sentences = corpus

    trigram_counts = defaultdict(Counter)
    bigram_counts = Counter()

    for sentence in sentences:
        tokens = tokenize(sentence)
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i + 1], tokens[i + 2]
            trigram_counts[(w1, w2)][w3] += 1
            bigram_counts[(w1, w2)] += 1

    trigram_probabilities = {}
    for bigram in trigram_counts:
        total = bigram_counts[bigram]
        trigram_probabilities[bigram] = {
            word: count / total for word, count in trigram_counts[bigram].items()
        }

    return trigram_probabilities

def local_surprisal(phrase:str, trigram_dict:Dict):
    """
        quantify surprisal locally based on the training dataset
        let's do trigram probabilities to be fancy unless it takes too long. 
    """
    probs = trigram_dict
    if probs is None:
        probs = generate_trigram_probabilities()
    tokenized_phrase = tokenize(phrase)
    
    ...
def model_based_surprisal():
    """
        quantify suprisal using probabilities from a large model
        I would argue that this is ok because such labeling is allowed in POS tagging etc. 
    """
    ...