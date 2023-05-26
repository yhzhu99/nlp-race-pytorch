from typing import Dict
from pathlib import Path
import json
from functools import partial
from collections import OrderedDict
from argparse import ArgumentParser

import lineflow as lf
from transformers import AlbertForMultipleChoice, AutoTokenizer
from datasets import load_dataset, load_from_disk
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, Dataset

import logging
logging.disable(logging.WARNING)


class RACEDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['article'] + ' ' + item['question']
        choices = item['options']
        encoding = self.tokenizer([prompt] * len(choices), choices, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        label = ord(item['answer']) - ord('A')  # convert A, B, C, D to 0, 1, 2, 3
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label)
        }
    def __len__(self):
        return len(self.data)

class DataModule(L.LightningDataModule):
    def __init__(self, datadir, batch_size):
        super().__init__()
        dataset = load_from_disk(datadir)
        tokenizer = AutoTokenizer.from_pretrained("albert-base-v2", cache_dir="logs/cache/albert-tokenizer")

        train_data = RACEDataset(dataset['train'], tokenizer)
        val_data = RACEDataset(dataset['validation'], tokenizer)
        test_data = RACEDataset(dataset['test'], tokenizer)
        self._train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self._val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        self._test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    def train_dataloader(self):
        return self._train_dataloader
    def val_dataloader(self):
        return self._val_dataloader
    def test_dataloader(self):
        return self._test_dataloader

class Pipeline(L.LightningModule):

    def __init__(self):
        super().__init__()
        model = AlbertForMultipleChoice.from_pretrained("albert-base-v2", cache_dir="logs/cache/albert-model")
        self.model = model
        self.outputs = []

    def training_step(self, batch, batch_idx):
        labels = batch["label"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        model_outs = self.model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels
                )
        loss = model_outs.loss
        logits = model_outs.logits
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        model_outs = self.model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels
                )
        loss = model_outs.loss
        logits = model_outs.logits

        labels_hat = torch.argmax(logits, dim=1)
        correct_count = torch.sum(labels == labels_hat)

        output = OrderedDict({
                "val_loss": loss,
                "correct_count": correct_count,
                "batch_size": len(labels)
                })
        self.outputs.append(output)

        return output

    def on_validation_epoch_end(self):
        val_acc = sum([out["correct_count"] for out in self.outputs]) / sum(out["batch_size"] for out in self.outputs)
        val_loss = sum([out["val_loss"] for out in self.outputs]) / len(self.outputs)
        self.log('val_loss', val_loss, sync_dist=True)
        self.log('val_acc', val_acc, sync_dist=True)        
        self.outputs.clear()
        return val_loss

    def test_step(self, batch, batch_idx):
        labels = batch["label"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        model_outs = self.model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels
                )
        loss = model_outs.loss
        logits = model_outs.logits

        labels_hat = torch.argmax(logits, dim=1)
        correct_count = torch.sum(labels == labels_hat)

        output = OrderedDict({
                "test_loss": loss,
                "correct_count": correct_count,
                "batch_size": len(labels)
                })
        self.outputs.append(output)

        return output

    def on_test_epoch_end(self):
        test_acc = sum([out["correct_count"] for out in self.outputs]) / sum(out["batch_size"] for out in self.outputs)
        test_loss = sum([out["test_loss"] for out in self.outputs]) / len(self.outputs)
        self.log('test_loss', test_loss, sync_dist=True)
        self.log('test_acc', test_acc, sync_dist=True)        
        self.outputs.clear()
        return test_loss

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        weight_decay = 0.0
        adam_epsilon = 1e-8

        # optimizer_grouped_parameters = [
        #     {
        #         'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
        #         'weight_decay': weight_decay
        #         },
        #     {
        #         'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
        #         'weight_decay': 0.0,
        #         }
        #     ]
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-6, eps=adam_epsilon)

        return optimizer

if __name__ == "__main__":
    L.seed_everything(42)
    config = {
        "batch_size": 8,
        }

    early_stop_callback = EarlyStopping(
            monitor="val_acc",
            min_delta=0.0,
            patience=1,
            verbose=True,
            mode="max",
            )

    checkpoint_filename = 'albert-all'
    logger = CSVLogger(save_dir="logs", name=f'race', version=checkpoint_filename)
    checkpoint_callback = ModelCheckpoint(filename="best", monitor="val_acc", mode="max")
    dm = DataModule(datadir=f"datasets_hf/race_all", batch_size=config["batch_size"])
    pipeline = Pipeline()
    trainer = L.Trainer(
            accelerator="gpu", devices=[0],
            max_epochs=5,
            logger=logger,
            callbacks=[early_stop_callback, checkpoint_callback],
            )
    # trainer.fit(pipeline, dm)
    trainer.test(pipeline, dm, ckpt_path="logs/race/albert-all/checkpoints/best.ckpt")
