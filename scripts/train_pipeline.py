
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import os
from datasets import Dataset, load_dataset
import math

def train_model(mlm_prob: float = 0.15):
    tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
    configuration = RobertaConfig()
    model = RobertaForMaskedLM(configuration)

    tokenizer.pad_token = tokenizer.eos_token
    data_folder = ""
    def preprocess_function(examples):
        return tokenizer(examples["text"])
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_prob)
    for curricula in os.listdir(data_folder):
        lm_dataset = load_dataset("text", data_files={"test": "data/{curricula}/test/test.test"}) # TODO: create train, val, test files that are a combo of all of the files in the data split 
        lm_dataset = lm_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=4,
        )

        training_args = TrainingArguments(
            output_dir="curriculum_learning",
            eval_strategy="epoch",
            learning_rate=2e-5,
            num_train_epochs=10, # TODO: increase
            weight_decay=0.01,
            push_to_hub=False,
            save_strategy="epoch"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=lm_dataset["train"],
            eval_dataset=lm_dataset["val"], #TODO uncomment
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        trainer.train()
    eval_results = trainer.evaluate()
    model.save_model("curriculum_learning")
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

if __name__ == "__main__":
    train_model()
