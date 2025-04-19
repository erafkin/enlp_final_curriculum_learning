from collections import defaultdict, Counter
import re
from typing import Dict
import math
import pandas as pd
from tqdm import tqdm
import pickle

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

    for sentence in tqdm(sentences):
        tokens = tokenize(sentence)
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i + 1], tokens[i + 2]
            trigram_counts[(w1, w2)][w3] += 1
            bigram_counts[(w1, w2)] += 1

    trigram_probabilities = {}
    for bigram in tqdm(trigram_counts):
        total = bigram_counts[bigram]
        trigram_probabilities[bigram] = {
            word: count / total for word, count in trigram_counts[bigram].items()
        }
    with open("trigram_probabilities.pkl", 'wb') as file:
        pickle.dump(trigram_probabilities, file)
        file.close()

    return trigram_probabilities

def local_surprisal(phrase:str, trigram_dict:Dict):
    """
        quantify surprisal locally based on the training dataset
        let's do trigram probabilities to be fancy unless it takes too long. 
    """
    tokenized_phrase = tokenize(phrase)
    probs = []
    for i in range(len(tokenized_phrase) - 2):
        w1, w2, w3 = tokenized_phrase[i], tokenized_phrase[i + 1], tokenized_phrase[i + 2]
        if (w1, w2)in trigram_dict and w3 in trigram_dict[(w1, w2)]:
            probs.append(trigram_dict[(w1, w2)][w3])
        
    surprisal = [-math.log2(prob) for prob in probs]
    phrase_surprisal = sum(surprisal)
    return phrase_surprisal

def model_based_surprisal():
    """
        quantify suprisal using probabilities from a large model
        I would argue that this is ok because such labeling is allowed in POS tagg/ing etc. 
    """
    ...

def main():
    surprisal_mode = "local" # "local" or "model"
    pickle_file = "./trigram_probabilities.pkl"
    train = []
    dev = []
    print("loading training corpus")
    with open("./data/train_100M/train.train", "r") as f:
        train = f.readlines()
        f.close()
    print("loading dev corpus")
    with open("./data/dev/dev.dev", "r") as f:
        dev = f.readlines()
        f.close()

    if surprisal_mode == "local":
        print("loading trigram probs")
        if pickle_file is None:
            train_trigram_probability_table = generate_trigram_probabilities(train)
        else:
            with open(pickle_file, 'rb') as file:
                train_trigram_probability_table = pickle.load(file)
                file.close()
        print("calculating train surprisal")
        train_rows = [[phrase, local_surprisal(phrase, train_trigram_probability_table)] for phrase in tqdm(train)]
        print("calculating dev surprisal")
        dev_rows = [[phrase, local_surprisal(phrase, train_trigram_probability_table)] for phrase in tqdm(dev)]
        train_df = pd.DataFrame(train_rows, columns=["phrase", "surprisal"])
        train_df.to_csv("train_local_surprisal.csv", index=False)
        dev_df = pd.DataFrame(dev_rows, columns=["phrase", "surprisal"])
        dev_df.to_csv("dev_local_surprisal.csv", index=False)
if __name__=="__main__":
    main()