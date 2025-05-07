from transformers import RobertaTokenizerFast, AutoModelForMaskedLM, RobertaConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import os
import json
from datasets import load_dataset
import math
# add the path to the custom data collator
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_processing.custom_data_collator import UsasTagAwareDataCollator

def train_model(mlm_prob: float = 0.15):
    tokenizer = RobertaTokenizerFast.from_pretrained("phueb/BabyBERTa-1")
    configuration = RobertaConfig.from_pretrained("phueb/BabyBERTa-1")
    model = AutoModelForMaskedLM.from_config(configuration)

    tokenizer.pad_token = tokenizer.eos_token
    data_folder = "data/aggregated"
    
    # Modified preprocess_function to manually add word_ids
    def preprocess_function(examples):
        tokenized_output = tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=128
        )
        
        # Manually get word_ids for each sequence in the batch
        word_ids_list = []
        for i in range(len(examples["text"])):
             word_ids_list.append(tokenized_output.word_ids(batch_index=i))

        # Add the list of word_ids to the output dictionary
        tokenized_output["word_ids"] = word_ids_list
        tokenized_output["text"] = examples["text"] # Keep text for custom collator
        return tokenized_output
    
    for curricula in ["level_1", "level_2", "level_3", "level_4", "level_5"]:
        # Load the dataset *before* deciding on the collator
        lm_dataset = load_dataset("text", data_files={"train": f"{data_folder}/train.train", "val":f"{data_folder}/dev.dev"}) 
        lm_dataset = lm_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=4,
        )

        if curricula == "level_5":
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_prob)
            # Remove columns incompatible with the default collator *after* mapping
            columns_to_remove = [col for col in ["text", "word_ids"] if col in lm_dataset["train"].column_names]
            if columns_to_remove:
                print(f"Level 5: Removing columns {columns_to_remove} for default collator.")
                lm_dataset = lm_dataset.remove_columns(columns_to_remove)
            num_train_epochs = 1
        else:
            # Load weights from the JSON configuration file
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                    "data_processing", f"{curricula}_tag_weights_config.json")
            try:
                with open(config_path, 'r') as f:
                    weights_config = json.load(f)
                pos_tag_weights = weights_config.get("pos_tag_weights", {})
                sem_tag_weights = weights_config.get("sem_tag_weights", {})
                print(f"Loaded tag weights from: {config_path}")
            except Exception as e:
                print(f"Warning: Could not load tag weights from {config_path}: {e}")
                print("Using default weights instead.")
                pos_tag_weights = {}
                sem_tag_weights = {}
            
            data_collator = UsasTagAwareDataCollator(
                tokenizer=tokenizer, 
                mlm_probability=mlm_prob,
                pos_tag_weights=pos_tag_weights,
                sem_tag_weights=sem_tag_weights,
                spacy_model="en_core_web_sm"
            )
            num_train_epochs = 2

        training_args = TrainingArguments(
            output_dir="curriculum_learning_reverse",
            eval_strategy="epoch",
            learning_rate=2e-5,
            num_train_epochs=num_train_epochs,
            weight_decay=0.01,
            push_to_hub=False,
            save_strategy="epoch",
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            remove_unused_columns=False
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=lm_dataset["train"],
            eval_dataset=lm_dataset["val"],
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        trainer.train()
        trainer.save_model(f"curriculum_learning_reverse/{curricula}")
        model = trainer.model
    eval_results = trainer.evaluate()
    trainer.save_model("curriculum_learning_reverse/final")
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

if __name__ == "__main__":
    train_model()
