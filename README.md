# ENLP Final Curriculum Learning
For our final project we decided to explore the impact of curriculum learning for the BabyLM challenge. We selected different methods for creating our curricula and finetuned a [BabyBERTa](https://huggingface.co/phueb/BabyBERTa-1) model on increasingly more difficult data. The three methods we chose are: 

1. **Surprisal**
    Surprisal theory posits that the amount of information contained in each word can be measured using surprisal (negative log probability). Studies on surprisal theory have shown that surprisal is a good predictor for longer reading and parsing times. This implies that surprisal is correlated with sentence difficulty. Therefore, the corpus can be divided into curricula with the easiest curriculum containing the phrases with the lowest surprisal. We will use surprisal calculated from trigram probabilities from the training set. 
2. **Syntactic**
    Syntactic complexity ordering, as a concept, is essentially about ranking sentences based on how complex their grammatical structure is. The data is typically sorted using a complexity score so that simpler sentences are used earlier in training and more complex sentences appear later in training. We will use spaCy to analyse the sentence structure in the training corpus for this approach

3. **MMM Theory**
    Maximize Minimal Means (MMM) is a curriculum learning strategy grounded in linguistic theories of language acquisition, which suggest that learners benefit from progressing from simpler to more complex structures. It does so by ranking sentences based on the average rarity of their syntactic (e.g., POS tags) and semantic (e.g., semantic role labels) features. Simpler, more frequent constructions are introduced earlier, while rarer, complex ones appear later. This mirrors how humans typically acquire language.
4. **Concreteness**
    The concreteness of a word ranges from a maximum of 5 to a minimum of 1 based on the Brybaert 2014 concreteness study. Concreteness represents the extent to which you can directly experience through your senses or actions like smelling, tasting, touching, hearing. A rating of 5 means that it is something that exists in reality. A rating of 1 means that it is something that you can’t experience directly. Based on the ratings from the study, we will assign a concreteness score to each sentence in the training dataset, meaning computing the average concreteness score of the words in the sentence. And based on the ranking, we will divide the dataset into 4 curricula where curriculum 1 contains the sentences with the highest concreteness score (considered as easiest) and curriculum 4 contains the sentences with the lowest concreteness score (considered as most complex).


## Information
This project was developed in python3.10.

### Setup
Setup virtual environment, e.g. on a mac:
```bash
python3.10 -m venv venv
source venv/bin/activate
```

Then install dependencies:
```bash
pip install -r requirements.txt
```
### Run 
To generate data, download the BabyLM dataset and consolidate all of the seperate data files into a single data file for each split called `split.split`--e.g. `train.train`, `dev.dev`, `test.test`.

Then depending on which curriculum learning method you would like to train you model with, the `data_processing` folder contains scripts to split these data files into curricula. 
- **Surprisal**: In order to split the data into curricula based on surprisal run `data_processing/surprisal_cl.py`. Note that it is currently set to use trigram probabilities calculated form the training data, but you can also swap out `surprisal_mode` to be `model` to retrieve surprisal scores for each input sentence based on BERT probabilities. Unfortunately resource and time constraints limited our ability to run the pipeline using BERT.
 

To run the pipeline set the `data_folder` variable to the appropriate path and then run `python scripts/train_pipeline.py`. If you are using a VM that uses slurm to manage jobs, there is a training slurm script in `scripts/slurm`.

## Compare
We saved off our models and curricula (zipped) as well as our evaluations in the `evaluation_results` folder. We evaluated our models using the [BabyLM 2024 evaluation pipeline](https://github.com/babylm/evaluation-pipeline-2024)

## Authors
- Saim Ishtiaq
- Emma Rafkin
- Ismail Shaheen
- Muxiang Wen 
