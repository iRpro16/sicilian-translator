from datasets import load_dataset
from transformers import T5ForConditionalGeneration, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
import numpy as np

class Trainer():
    def __init__(
            self,
            new_tokenizer: object,
            data_files: dict,
            dataset_type: str,
            max_length: int,
            model
    ) -> None:
        self.new_tokenizer = new_tokenizer
        self.data_files = data_files
        self.dataset_type = dataset_type
        self.model = model
        self.max_length = max_length

    def configure(self):
        ## load dataset
        dataset = load_dataset(self.dataset_type, data_files=self.data_files)
        ## remove column
        dataset = dataset.remove_columns(["Unnamed: 0"])
        ## configure model embeddings
        self.model.resize_token_embeddings(len(self.new_tokenizer))
        return dataset
    
    # Function to use for preprocessing
    def preprocess_function(self, examples):
        inputs = [ex for ex in examples["english"]]
        targets = [ex for ex in examples["sicilian"]]
        model_inputs = self.new_tokenizer(inputs, text_target=targets, max_length=self.max_length,
                                      truncation=True)
        return model_inputs
    
    # Preproccess the data and tokenize it
    def preprocess_dataset(self):
        dataset = self.configure()
        tokenized_datasets = dataset.map(self.preprocess_function, batched=True, desc="Running tokenizer",
                                         remove_columns=dataset["train"].column_names)
        return tokenized_datasets
    
    # Function to compute loss metric
    def compute_metrics(self, eval_preds):
        metric = evaluate.load("sacrebleu")
        preds, labels = eval_preds
        ## in case model returns more than prediction logits
        if isinstance(preds, tuple):
            preds=preds[0]
        decoded_preds = self.new_tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100s in the labels as we can't decode them
        labels = np.where(labels != -100, labels, self.new_tokenizer.pad_token_id)
        decoded_labels = self.new_tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        # result
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]} 
    
class Finetune():
    def __init__(self, model_dir: str, batch_size: int):
        self.model_dir = model_dir
        self.batch_size = batch_size

    # Args for finetuning
    def args(self):
        return Seq2SeqTrainingArguments(
            self.model_dir,
            eval_strategy="no",
            save_strategy="epoch",
            learning_rate=1e-3,
            num_train_epochs=3,
            save_total_limit=3,
            predict_with_generate=True,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=64,
            fp16=True,
            #push_to_hub=True
        )
    
    # Finetune the model
    def finetune_model(self, tokenizer: object, data_files: dict, dataset_type: str, 
                       max_length: int, model):
        ## get trainer to get and preprocess data
        trainer = Trainer(tokenizer, data_files, dataset_type, max_length, model)
        tokenized_datasets = trainer.preprocess_dataset()
        ## data collator for Seq2SeqTrainer
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        return Seq2SeqTrainer(
            model=trainer.model,
            args=self.args(),
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=trainer.compute_metrics
        )