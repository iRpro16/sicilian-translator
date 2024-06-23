import re
import pandas as pd

path = "/home/irpro16/projects/sicilian-translator/datasets/multi-dataset"

class Preprocess():
    def __init__(self, tokenizer_dataset):
        self.tokenizer_dataset = pd.read_csv(tokenizer_dataset)

    def normalize(self, corpus):
        ## lowercase text
        text_lower = corpus.lower()

        ## remove brackets and parentheses
        text_clean = text_lower.replace('(','').replace(')','').replace('[','').replace(']','').replace(':','')

        ## remove numbers
        text_alpha = re.sub(r'[0-9]', '',text_clean)

        return text_alpha
    
    def apply_normalization(self):
        clean_corpus = self.tokenizer_dataset['0'].apply(self.normalize)
        return clean_corpus