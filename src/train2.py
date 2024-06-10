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

## ddp_setup
def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

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

class Trainer:
    def __init__(
            self, 
            model,
            train_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            gpu_id: int,
            save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _run_batch(self, source, targets, attention_mask, decoder_attenion_mask):
        self.optimizer.zero_grad()
        outputs = self.model(input_ids = source,
                             decoder_input_ids = targets,
                             attention_mask = attention_mask,
                             decoder_attenion_mask= decoder_attenion_mask
                             )
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        logits = outputs.logits
        loss = loss_fn(logits.view(-1, 530482), targets.contiguous().view(-1))
        loss.backward()
        self.optimizer.step()
        return loss.item()
        
    def _run_epoch(self):
        losses = 0
        for batch in tqdm(self.train_data, total=len(list(self.train_data))):
            src = batch["input_ids"].to(self.gpu_id)
            tgt = batch["decoder_input_ids"].to(self.gpu_id)
            attention_mask = batch["attention_mask"].to(self.gpu_id)
            decoder_attention_mask = batch["decoder_attention_mask"].to(self.gpu_id)
            loss_item = self._run_batch(src, tgt, attention_mask, decoder_attention_mask)
            losses += loss_item
        return losses / len(list(self.train_data))

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        torch.save(ckp, "checkpoint.pt")
        print(f"Epoch {epoch} | Training checkpoint saved at checkpoint")

    def train(self, max_epochs: int):
        train_loss_list = []
        for epoch in range(max_epochs):
            start_time = timer()
            train_loss = self._run_epoch(epoch)
            train_loss_list.append(train_loss)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
            end_time = timer()
            print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s \n"))


def load_train_objs():
    train_set = TokenData(train_dataset, tokenizer=tokenizer)
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    model.resize_token_embeddings(len(tokenizer)) # resize model embeddings
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, betas=(0.9,0.98), eps = 1e-9)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(dataset)
        )
    
def main(rank: int,world_size: int, total_epochs: int, save_every: int):
        ddp_setup(rank, world_size)
        dataset, model, optimizer = load_train_objs()
        train_data = prepare_dataloader(dataset, batch_size=32)
        trainer = Trainer(model, train_data, optimizer, rank, save_every)
        trainer.train(total_epochs)
        destroy_process_group()

if __name__ == "__main__":
    import sys
    total_epochs = int(sys.argv[1])
    save_every = int(sys.argv[2])
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, total_epochs, save_every), nprocs=world_size)