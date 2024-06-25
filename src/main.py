# Import modules
from src.data.make_dataset import Dataset
from src.tokenizer.create_tokenizer import CreateTokenizer
from src.features.build_features import Preprocess
from src.models.train_model import Trainer, Finetune
from transformers import AutoTokenizer, T5ForConditionalGeneration

# Dataset paths
path_english = "/mnt/c/Users/Isidoro/Documents/datasets/en-si/english_texts.txt"
path_sicilian = "/mnt/c/Users/Isidoro/Documents/datasets/en-si/sicilian_texts.txt"

# Tokenizer
old_tokenizer = AutoTokenizer.from_pretrained("t5-small")


if __name__ == "__main__":
    # Get the Dataset
    fetch_datasets = Dataset(path_sicilian, path_english)
    # Clean datasets
    ds_english, ds_sicilian = fetch_datasets.clean_dataset()
    # Create tokenizer dataset and dataset for splitting
    train_test_ds, tokenizer_ds = fetch_datasets.create_datasets(ds_english, ds_sicilian)
    # Train test split
    train_ds, test_ds = fetch_datasets.split_dataset(train_test_ds)

    ## Save datasets using fetch_datasets.save_datasets() method

    # Path
    multi_dataset = "/home/irpro16/projects/sicilian-translator/datasets/multi-dataset"

    # Preprocess dataset for tokenizer, class expects CSV
    preprocess = Preprocess(multi_dataset)
    # Clean dataset
    clean_corpus = preprocess.apply_normalization()

    ## Save clean corpus to train the tokenizer
    
    # Train tokenizer
    data_file = {"train": "/home/irpro16/projects/sicilian-translator/datasets/clean-multi-dataset"}
    create_tokenizer = CreateTokenizer(file_type="csv", data_file=data_file, 
                                       old_tokenizer=old_tokenizer)
    tokenizer = create_tokenizer.train_tokenizer(old_tokenizer=old_tokenizer, vocab_size=52000)

    # Finetune model by getting saved dataset and tokenizer from before
    train_ds = "/home/irpro16/projects/sicilian-translator/datasets/train_dataset"
    test_ds = "/home/irpro16/projects/sicilian-translator/datasets/test_dataset"
    tokenizer_path = "/home/irpro16/projects/sicilian-translator/notebooks/tokenizer-small"
    # Get pretrained model
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, return_tensors="pt")
    data_files = {"train": train_ds, "test": test_ds}
    # Start the training
    finetune = Finetune("model_trainer", batch_size=32)
    train_model = finetune.finetune_model(
        tokenizer=tokenizer,
        data_files=data_files,
        dataset_type="csv",
        max_length=52,
        model=model
    )
    
    # train_model.train()