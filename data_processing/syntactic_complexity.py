from collections import defaultdict, Counter
import os
import pickle

import spacy
import numpy as np
import pandas as pd
from typing import List
from tqdm import tqdm

print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat", "lemmatizer"])

def calculate_complexity(doc) -> float:
    if len(doc)==0:
        return 0.0

    try:
        tree_depth = max([abs(token.head.i - token.i) for token in doc if token.head != token])
    except ValueError:
        tree_depth = 0
    num_clauses = len([token for token in doc if token.dep_ in ["ccomp", "xcomp", "advcl", "acl", "relcl"]])
    word_lengths = [len(token.text) for token in doc if not token.is_punct and not token.is_space]
    avg_word_length = np.mean(word_lengths) if word_lengths else 0
    sentence_length = len([token for token in doc if not token.is_space])
    complexity_score = (0.3*tree_depth)+(0.4*num_clauses)+(0.1*avg_word_length)+(0.2*sentence_length)
    return complexity_score

def batch_calculate_complexity(texts: List[str], batch_size: int = 500, save_path: str = None) -> List[float]:
    complexity_scores = []
    start_index = 0
    if save_path and os.path.exists(save_path):
        print(f"Loading partial results from {save_path}")
        with open(save_path, 'rb') as f:
            saved_data = pickle.load(f)
            complexity_scores = saved_data.get('complexity_scores', [])
            start_index = saved_data.get('current_index', 0)
    print(f"Starting batch processing from index {start_index}...")
    for i in tqdm(range(start_index, len(texts), batch_size), desc="Calculating syntactic complexity"):
        batch = texts[i:i+batch_size]
        docs = list(nlp.pipe(batch, disable=["ner", "textcat", "lemmatizer"]))
        batch_scores = [calculate_complexity(doc) for doc in docs]
        complexity_scores.extend(batch_scores)
        if save_path and (i // batch_size) % 10 == 0:
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'complexity_scores': complexity_scores,
                    'current_index': i + batch_size
                }, f)
            print(f"Checkpoint saved at index {i}")
    return complexity_scores

def create_curricula(df: pd.DataFrame, split: str = "train", output_dir: str = "./syntactic_curricula"):
    os.makedirs(f"{output_dir}/easy/{split}", exist_ok=True)
    os.makedirs(f"{output_dir}/medium/{split}", exist_ok=True)
    os.makedirs(f"{output_dir}/hard/{split}", exist_ok=True)
    phrases = df.phrase.to_list()
    scores = df.complexity.to_list()
    zipped_list = list(zip(phrases, scores))
    zipped_list = [z for z in zipped_list if z[0].strip() != ""]
    zipped_list.sort(key=lambda x: x[1])
    easy_idx = int(len(zipped_list)*(1/3))
    medium_idx = int(len(zipped_list) * (2/3))
    print(f"Creating {split} curricula:")
    print(f"  Easy: {easy_idx} phrases")
    print(f"  Medium: {medium_idx} phrases")
    print(f"  Hard: {len(zipped_list)} phrases")

    with open(f"{output_dir}/easy/{split}/{split}.{split}", "w") as f:
        f.write("".join([z[0] for z in zipped_list[:easy_idx]]))
    with open(f"{output_dir}/medium/{split}/{split}.{split}", "w") as f:
        f.write("".join([z[0] for z in zipped_list[:medium_idx]]))
    with open(f"{output_dir}/hard/{split}/{split}.{split}", "w") as f:
        f.write("".join([z[0] for z in zipped_list]))


def main():
    print("Loading training corpus...")
    with open("./data/train_10M/train.train", "r", encoding="utf-8", errors="ignore") as f:
        train = f.readlines()
    print("loading dev corpus...")
    with open("./data/dev/dev.dev", "r", encoding="utf-8", errors="ignore") as f:
        dev = f.readlines()
    pickle_file = "./syntactic_complexity_scores.pkl"
    if os.path.exists(pickle_file):
        print("Loading pre-calculated complexity scores...")
        with open(pickle_file, 'rb') as file:
            saved_data = pickle.load(file)
            train_scores = saved_data['train_scores']
            dev_scores = saved_data['dev_scores']
    else:
        print("Calculating syntactic complexity scores...")
        train_scores = batch_calculate_complexity(train, save_path="./train_checkpoint.pkl")
        dev_scores = batch_calculate_complexity(dev, save_path="./dev_checkpoint.pkl")
        with open(pickle_file, 'wb') as file:
            pickle.dump({
                'train_scores': train_scores,
                'dev_scores': dev_scores
            }, file)
    print("Creating DataFrames...")
    train_df = pd.DataFrame(zip(train, train_scores), columns=["phrase", "complexity"])
    dev_df = pd.DataFrame(zip(dev, dev_scores), columns=["phrase", "complexity"])
    train_df.to_csv("train_syntactic_complexity.csv", index=False)
    dev_df.to_csv("dev_syntactic_complexity.csv", index=False)
    print("Creating curriculum splits...")
    create_curricula(train_df, split="train")
    create_curricula(dev_df, split="dev")
    print("Completed! Curriculum files saved")

if __name__ == "__main__":
    main()