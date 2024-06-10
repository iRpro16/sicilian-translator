import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from timeit import default_timer as timer
from collections import OrderedDict
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

device = "cuda" if torch.cuda.is_available() else "cpu"

## path to csv dataset
path_dataset = "train_dataset"

## fetch tokenizer
path_token = "tokenizer"

## TokenDataset
class TokenData(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.datframe = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.datframe)

    def __getitem__(self, idx):
        src_txt = self.datframe['english'].iloc[idx]
        tgt_txt = self.datframe['sicilian'].iloc[idx]
        preprocessed_src = self.tokenizer(src_txt, padding='max_length',
                                          truncation=True, max_length=52)
        preprocessed_tgt = self.tokenizer(tgt_txt, padding='max_length',
                                          truncation=True, max_length=52)
        return {
            "input_ids": torch.tensor(preprocessed_src["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(preprocessed_src["attention_mask"], dtype=torch.long),
            "decoder_input_ids": torch.tensor(preprocessed_tgt["input_ids"], dtype=torch.long),
            "decoder_attention_mask": torch.tensor(preprocessed_tgt["attention_mask"], dtype=torch.long)
        }

## Fetch the data
class FetchData:
    def __init__(self, token_path, dataset_path):
        self.token_path = token_path
        self.dataset_path = dataset_path

    def fetch(self):
        train_dataset = pd.read_csv(self.dataset_path)
        tokenizer = T5Tokenizer.from_pretrained(self.token_path)

        ## drop column
        train_dataset.drop('Unnamed: 0', axis=1, inplace=True)

        return train_dataset, tokenizer
    

## fetch data and tokenizer
data_fetcher = FetchData(path_token, path_dataset)
train_dataset, tokenizer = data_fetcher.fetch()

## preprocessed dataset
preprocessed_train = TokenData(train_dataset, tokenizer=tokenizer)
print(preprocessed_train[1])

## train dataloader
train_dataloader = DataLoader(
    preprocessed_train, 
    batch_size=32,
    pin_memory=True,
    shuffle=False,
    sampler=DistributedSampler(preprocessed_train)
    )

## model
model = T5ForConditionalGeneration.from_pretrained("t5-base")
model.resize_token_embeddings(len(tokenizer)) # resize model embeddings

## loss function
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, betas=(0.9,0.98), eps = 1e-9)
num_epochs = 10

## model training
def train_epoch(model, optimizer):
    print('Training...')
    losses = 0
    model.train()

    for batch in tqdm(train_dataloader, total=len(list(train_dataloader))):

        # src and target
        src = batch["input_ids"].to(device)
        tgt = batch["decoder_input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)

        # passing data to model
        outputs = model(
            input_ids=src, 
            decoder_input_ids=tgt,
            attention_mask=attention_mask, 
            decoder_attention_mask=decoder_attention_mask)
        
        # gradients to zero
        optimizer.zero_grad()
        
        # logits
        logits = outputs.logits

        # loss
        loss = loss_fn(logits.view(-1, 530482), tgt.contiguous().view(-1))

        # calculating gradient for the loss function
        loss.backward()

        # optimizing the running loss for logging purposes
        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_dataloader))

## train
train_loss_list = []
for epoch in range(1, num_epochs+1):
    start_time = timer()
    train_loss = train_epoch(model, optimizer)
    end_time = timer()
    train_loss_list.append(train_loss)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s \n"))
