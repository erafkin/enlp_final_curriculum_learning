from collections import defaultdict, Counter
import re
from typing import Dict
import math
import pandas as pd
from tqdm import tqdm
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 
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

def model_based_surprisal(model, tokenizer, phrase):
    """
        quantify suprisal using probabilities from a large model
        I would argue that this is ok because such labeling is allowed in POS tagg/ing etc. 
    """
    with torch.no_grad():
        input = tokenizer(phrase, return_tensors="pt")
        generated_outputs = model(input.input_ids, output_scores=True)
        probs = [torch.nn.functional.softmax(generated_outputs.logits[0, i, :], dim = 0)[input_id].item() for i, input_id in enumerate(input.input_ids[0])]
    surprisal = [-math.log2(prob) for prob in probs]
    phrase_surprisal = sum(surprisal)
    return phrase_surprisal
def create_curricula(df: pd.DataFrame, split: str = "train"):
    """
        Split dataframes into easy, medium, hard curricula
    """
    phrases = df.phrase.to_list()
    scores = df.surprisal.to_list()
    zipped_list = list(zip(phrases, scores))
    zipped_list.sort(key=lambda x: x[1])
    zipped_list = [z for z in zipped_list if z[0].strip() != ""]
    easy_idx = int(len(zipped_list) * (1/3))
    medium_idx = int(len(zipped_list) * (2/3))
    with open(f"./surprisal_curricula_local/easy/{split}.{split}", "w") as f:
        easy_str = "".join([z[0] for z in zipped_list[:easy_idx]])
        f.write(easy_str)
        f.close()
    with open(f"./surprisal_curricula_local/medium/{split}.{split}", "w") as f:
        medium_str = "".join([z[0] for z in zipped_list[:medium_idx]])
        f.write(medium_str)
        f.close()
    with open(f"./surprisal_curricula_local/hard/{split}.{split}", "w") as f:
        hard_str = "".join([z[0] for z in zipped_list])
        f.write(hard_str)
        f.close()

def main(surprisal_mode:str = "local"):
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
        pickle_file = "./trigram_probabilities.pkl" # I already generated this and saved it off locally, set to None if you need to regenerate
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
    else:
        model = AutoModelForCausalLM.from_pretrained("gpt2", return_dict_in_generate=True)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        print("calculating dev surprisal")
        dev_rows = [[phrase,  model_based_surprisal(model, tokenizer,phrase)] for phrase in tqdm(dev)]
        dev_df = pd.DataFrame(dev_rows, columns=["phrase", "surprisal"])
        dev_df.to_csv("dev_model_surprisal.csv", index=False)
        print("calculating train surprisal")
        train_rows = [[phrase, model_based_surprisal(model, tokenizer,phrase)] for phrase in tqdm(train)]
        train_df = pd.DataFrame(train_rows, columns=["phrase", "surprisal"])
        train_df.to_csv("train_model_surprisal.csv", index=False)
if __name__=="__main__":
    # uncomment this to calculate dataframes of surprisal for the splits
    # surprisal_mode = "model" # "local" or "model
    # main(surprisal_mode=surprisal_mode)

    # uncomment this code to actually split the data up into curricula based on surprisal

    train_df = pd.read_csv("./curriculum_learning/train_local_surprisal.csv")
    create_curricula(train_df, split="train")
    dev_df = pd.read_csv("./curriculum_learning/dev_local_surprisal.csv")
    create_curricula(dev_df, split="dev")
