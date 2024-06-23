from sklearn.model_selection import train_test_split
import pandas as pd
import os

# Paths
path_english = "/mnt/c/Users/Isidoro/Documents/datasets/en-si/english_texts.txt"
path_sicilian = "/mnt/c/Users/Isidoro/Documents/datasets/en-si/sicilian_texts.txt"

class Dataset:
    def __init__(self, txt_sicilian, txt_english):
        self.txt_sicilian = txt_sicilian
        self.txt_english = txt_english

    # clean the dataset
    def clean_dataset(self):
        ## read files
        df_english_full = pd.read_fwf(self.txt_english, header=None, delimeter=",")
        df_sicilian_full = pd.read_fwf(self.txt_sicilian, header=None, delimeter=",")

        ## remove NaNs
        df_english_full.dropna(axis=1, inplace=True)

        ## rename columns
        df_english_full.rename(columns={0: 'english'}, inplace=True)
        df_sicilian_full.rename(columns={0: 'sicilian'}, inplace=True)

        ## select specific number of rows
        df_english = df_english_full.iloc[:467000]
        df_sicilian = df_sicilian_full.iloc[:467000]

        return df_english, df_sicilian
    
    # create one dataset for train/test and one for tokenizer
    def create_datasets(self, df_english, df_sicilian):
        ## get datasets
        datasets = [df_english['english'], df_sicilian['sicilian']]
        
        ## dataset for train/test
        train_test_dataset = pd.concat(datasets, axis=1)

        ## dataset for tokenizer
        tokenizer_dataset = pd.concat(datasets)

        return train_test_dataset, tokenizer_dataset
    
    # split the dataset
    def split_dataset(self, train_test_dataset):
        train_dataset, test_dataset = train_test_split(train_test_dataset, train_size=0.9)
        return train_dataset, test_dataset
    
    # save a dataset
    def save_datasets(self, dataset, path_save: str):
        cwd = os.getcwd()
        path = cwd + path_save
        dataset.to_csv(path)