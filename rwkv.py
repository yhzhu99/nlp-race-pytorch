from typing import Dict
from pathlib import Path
import json
from functools import partial
from collections import OrderedDict
from argparse import ArgumentParser

import lineflow as lf
from transformers import AutoTokenizer, AutoModelForMultipleChoice, RwkvConfig, RwkvModel
from datasets import load_dataset, load_from_disk
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

import torch
from torch import nn
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
        # encoding = self.tokenizer([prompt] * len(choices), choices, return_tensors='pt')
        encoding = []
        lens = []
        for choice in choices:
            cur = self.tokenizer(prompt, choice, return_tensors='pt')['input_ids'].squeeze()
            if len(cur)>512:
                cur = cur[:512]
                lens.append(512)
            else:
                lens.append(len(cur))
                pad_size = 512 - cur.size(0)
                cur = torch.nn.functional.pad(cur, (0, pad_size))

            encoding.append(cur)

        label = ord(item['answer']) - ord('A')  # convert A, B, C, D to 0, 1, 2, 3
        return {
            'input_ids': torch.stack(encoding, dim=0),
            'label': torch.tensor(label),
            'lens': torch.tensor(lens),
        }
    def __len__(self):
        return len(self.data)

class DataModule(L.LightningDataModule):
    def __init__(self, datadir, batch_size):
        super().__init__()
        dataset = load_from_disk(datadir)
        tokenizer = AutoTokenizer.from_pretrained("sgugger/rwkv-430M-pile", cache_dir="logs/cache/rwkv-430M-pile-tokenizer")


        train_data = RACEDataset(dataset['train'], tokenizer)
        val_data = RACEDataset(dataset['validation'], tokenizer)
        test_data = RACEDataset(dataset['test'], tokenizer)
        self._train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self._val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        self._test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        # for i in self._train_dataloader:
        #     print(i['input_ids'][0].shape, i['input_ids'][1].shape, len(i['input_ids']))

    def train_dataloader(self):
        return self._train_dataloader
    def val_dataloader(self):
        return self._val_dataloader
    def test_dataloader(self):
        return self._test_dataloader

class Pipeline(L.LightningModule):

    def __init__(self):
        super().__init__()
        # self.backbone = RwkvModel.from_pretrained("sgugger/rwkv-430M-pile", cache_dir="logs/cache/rwkv-430M-pile-model")
        self.model = RwkvModel.from_pretrained("sgugger/rwkv-430M-pile", cache_dir="logs/cache/rwkv-430M-pile-model")
        self.head = nn.Linear(1024, 1)
        self.outputs = []
    def forward(self, input_ids, labels, lens):
        # print(input_ids.shape)
        # import pdb
        # pdb.set_trace()
        bs, num_choices, hid = input_ids.shape
        x = input_ids.view(-1, hid)
        lens = lens.view(-1)
        outputs = self.model(x).last_hidden_state
        last_outputs = outputs[torch.arange(outputs.size(0)), lens-1, :]
        last_outputs = last_outputs.view(bs, num_choices, -1)
        # pooled_output = last_outputs.mean(dim=-1)
        # print("pool", pooled_output.shape)
        # print(last_outputs.shape)
        out = self.head(last_outputs)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(out, labels.unsqueeze(-1))
        return loss, out


    def training_step(self, batch, batch_idx):
        labels = batch["label"]
        input_ids = batch["input_ids"]
        lens = batch["lens"]
        loss, logits = self(input_ids, labels, lens)
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
        # print(input_ids.shape)
        attention_mask = batch["attention_mask"]

        # print("Input id", input_ids.shape)
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
        adam_epsilon = 1e-8
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-6, eps=adam_epsilon)
        return optimizer

if __name__ == "__main__":
    L.seed_everything(42)
    config = {
        "batch_size": 2,
        }

    early_stop_callback = EarlyStopping(
            monitor="val_acc",
            min_delta=0.0,
            patience=1,
            verbose=True,
            mode="max",
            )

    checkpoint_filename = 'rwkv-all'
    logger = CSVLogger(save_dir="logs", name=f'race', version=checkpoint_filename)
    checkpoint_callback = ModelCheckpoint(filename="best", monitor="val_acc", mode="max")
    dm = DataModule(datadir=f"datasets_hf/race_all", batch_size=config["batch_size"])
    pipeline = Pipeline()
    trainer = L.Trainer(
            accelerator="gpu", devices=[1],
            max_epochs=5,
            logger=logger,
            callbacks=[early_stop_callback, checkpoint_callback],
            num_sanity_val_steps=0
            )
    trainer.fit(pipeline, dm)
    # trainer.test(pipeline, dm, ckpt_path="logs/race/rwkv-all/checkpoints/best.ckpt")
