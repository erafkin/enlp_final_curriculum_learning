
from transformers import RobertaTokenizerFast, AutoModelForMaskedLM, RobertaConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import os
from datasets import load_dataset
import math

def train_model(mlm_prob: float = 0.15):
    tokenizer = RobertaTokenizerFast.from_pretrained("phueb/BabyBERTa-1")
    configuration = RobertaConfig.from_pretrained("phueb/BabyBERTa-1")
    model = AutoModelForMaskedLM.from_config(configuration)

    tokenizer.pad_token = tokenizer.eos_token
    data_folder = "./syntactic_curricula" # Swapped out with the folder where my curricula data are saved.
    def preprocess_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_prob)
    for curricula in ["easy", "medium", "hard"]:
        lm_dataset = load_dataset("text", data_files={
            "train": f"{data_folder}/{curricula}/train/train.train", #This is the one small bit different from the original train_pipeline.py
            "val": f"{data_folder}/{curricula}/dev/dev.dev"
        })
        lm_dataset = lm_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=4,
        )

        training_args = TrainingArguments(
            output_dir="curriculum_learning",
            eval_strategy="epoch",
            learning_rate=2e-5,
            num_train_epochs=3, # increased to 3
            weight_decay=0.01,
            push_to_hub=False,
            save_strategy="epoch"
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
        trainer.save_model(f"curriculum_learning/{curricula}")
        model = trainer.model
    eval_results = trainer.evaluate()
    trainer.save_model("curriculum_learning/final")
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

if __name__ == "__main__":
    train_model()
