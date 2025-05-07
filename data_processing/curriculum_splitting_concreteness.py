import csv
import re
import os
from datasets import load_dataset
import spacy
nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])
def split_by_concreteness(split_name: str,
                          output_base: str = "data10M",
                          num_slices: int = 3):
    # 1. Load concreteness norms into a dict
    concreteness = {}
    with open("brysbaert_2014_concreteness.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row["Word"].lower()
            try:
                score = float(row["Conc.M"])
            except ValueError:
                continue
            concreteness[word] = score

    # Pull the 10M‐word BabyLM dataset splits
    ds = load_dataset("nilq/babylm-10M", split=split_name)

    # 3. Compute a sentence‐level concreteness score
    # TODO: Try different way of treating unknown words in the brysbaert_2014_concreteness.csv to see if it improves the result, here we skip all of them when computing the average concreteness score for a sentence
    # def compute_score(example):
    #     words = re.findall(r"\w+", example["text"].lower())
    #     scores = [concreteness[w] for w in words if w in concreteness]
    #     # default to 0.0 (most abstract) if no rated words
    #     avg = sum(scores) / len(scores) if scores else 0.0
    #     return {"concreteness": avg}
    def compute_score(example):
        doc = nlp(example["text"].lower())
        scores = []
        for token in doc:
            if not token.is_alpha:
                continue
            word = token.text
            lemma = token.lemma_
            if word in concreteness:
                scores.append(concreteness[word])
            elif lemma in concreteness:
                scores.append(concreteness[lemma])
        avg = sum(scores) / len(scores) if scores else 0.0
        return {"concreteness": avg}

    ds = ds.map(compute_score, batched=False)

    # 4. Sort ascending (most concrete first)
    ds = ds.sort("concreteness", reverse=True)

    # 5. Split into equal slices
    n = len(ds)
    slice_size = n // num_slices
    slices = {}
    for i in range(num_slices):
        start = i * slice_size
        end = (i + 1) * slice_size if i < num_slices - 1 else n
        slices[f"slice_{i+1}"] = ds.select(range(start, end))
        # #TODO comment
        # for j in range(100):
        #     print(f"slice{i+1}")
        #     print(slices[f"slice_{i+1}"][j]["text"])

    # 6. Save each slice to its own folder
    for name, subset in slices.items():
        out_folder = os.path.join(output_base, name)
        os.makedirs(out_folder, exist_ok=True)
        out_path = os.path.join(out_folder, f"{split_name}.txt")
        with open(out_path, "w", encoding="utf-8") as f_out:
            for ex in subset:
                # flatten internal newlines, one sentence per line
                f_out.write(ex["text"].replace("\n", " ") + "\n")
        print(f"Wrote {len(subset)} lines to {out_path}")

if __name__ == "__main__":
    split_by_concreteness("train", output_base="data10M", num_slices=3)
    split_by_concreteness("validation", output_base="data10M", num_slices=3)
