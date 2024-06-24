from datasets import load_dataset
from transformers import AutoTokenizer

old_tokenizer = AutoTokenizer.from_pretrained("t5-small")

class CreateTokenizer():
    def __init__(self, file_type: str, data_file: dict, old_tokenizer):
        self.file_type = file_type
        self.data_file = data_file
        self.tokenizer = old_tokenizer

    # Load and remove extra column
    def clean_dataset(self):
        load_data = load_dataset(self.file_type, data_files=self.data_file)
        
        ## Extra column got loaded, so we remove unnecessary column
        raw_dataset =load_data.remove_columns(["Unnamed: 0"])
        return raw_dataset
    
    # Get training corpus and turn into iterator
    def get_training_corpus(self):
        raw_dataset = self.clean_dataset()
        dataset_iter = raw_dataset['train']
        for start_idx in range(0, len(dataset_iter), 1000):
            samples = dataset_iter[start_idx : start_idx + 1000]
            yield samples['0']

    # Train the tokenizer 
    def train_tokenizer(self, vocab_size):
        traning_corpus = self.get_training_corpus()
        trained_tokenizer = old_tokenizer.train_new_from_iterator(traning_corpus, vocab_size)
        return trained_tokenizer
    
    # Save the tokenizer
    def save_tokenizer(self, tokenizer, tokenizer_name: str):
        tokenizer.save_pretrained(tokenizer_name)