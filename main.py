from typing import Dict
from pathlib import Path
import json
from functools import partial
from collections import OrderedDict
from argparse import ArgumentParser

import lineflow as lf
from transformers import BertForMultipleChoice, BertTokenizer, AdamW
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

import logging
logging.disable(logging.WARNING)


MAX_LEN = 128
NUM_LABELS = 4
label_map = {"A": 0, "B": 1, "C": 2, "D": 3}


def raw_samples_to_dataset(samples):
    datas = []
    for sample in samples:
        for idx in range(len(sample["answers"])):
            _id = sample["id"]
            _article = sample["article"]
            _answer = sample["answers"][idx]
            _options = sample["options"][idx]
            _question = sample["questions"][idx]

            data = {
                    "id": _id,
                    "article": _article,
                    "answer": _answer,
                    "options": _options,
                    "question": _question,
                    }
            datas.append(data)
    return lf.Dataset(datas)


def preprocess(tokenizer: BertTokenizer, x: Dict) -> Dict:

    choices_features = []

    option: str
    for option in x["options"]:
        text_a = x["article"]
        if x["question"].find("_") != -1:
            text_b = x["question"].replace("_", option)
        else:
            text_b = x["question"] + " " + option

        inputs = tokenizer.encode_plus(
                text_a,
                text_b,
                add_special_tokens=True,
                max_length=MAX_LEN
                )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1] * len(input_ids)

        pad_token_id = tokenizer.pad_token_id
        padding_length = MAX_LEN - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_id] * padding_length)

        assert len(input_ids) == MAX_LEN, "Error with input length {} vs {}".format(len(input_ids), MAX_LEN)
        assert len(attention_mask) == MAX_LEN, "Error with input length {} vs {}".format(len(attention_mask), MAX_LEN)
        assert len(token_type_ids) == MAX_LEN, "Error with input length {} vs {}".format(len(token_type_ids), MAX_LEN)

        choices_features.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            })

    labels = label_map.get(x["answer"], -1)
    label = torch.tensor(labels).long()

    return {
            "id": x["id"],
            "label": label,
            "input_ids": torch.tensor([cf["input_ids"] for cf in choices_features]),
            "attention_mask": torch.tensor([cf["attention_mask"] for cf in choices_features]),
            "token_type_ids": torch.tensor([cf["token_type_ids"] for cf in choices_features]),
            }


def get_dataloader(datadir: str, batch_size, cachedir: str = "./datasets/cached"):
    datadir = Path(datadir)
    cachedir = Path(cachedir)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    preprocessor = partial(preprocess, tokenizer)

    train_samples = []
    for grade in ("middle", "high"):
        for _path in (datadir / "train" / grade).iterdir():
            train_samples.append(json.loads(_path.read_text()))
    train = raw_samples_to_dataset(train_samples)
    train_dataloader = DataLoader(
            train.map(preprocessor).save(cachedir / "train.cache"),
            sampler=RandomSampler(train),
            batch_size=batch_size,
            num_workers=8,
            )

    val_samples = []
    for grade in ("middle", "high"):
        for _path in (datadir / "dev" / grade).iterdir():
            val_samples.append(json.loads(_path.read_text()))
    val = raw_samples_to_dataset(val_samples)
    val_dataloader = DataLoader(
            val.map(preprocessor).save(cachedir / "val.cache"),
            sampler=SequentialSampler(val),
            batch_size=batch_size,
            num_workers=8,
            )

    test_samples = []
    for grade in ("middle", "high"):
        for _path in (datadir / "test" / grade).iterdir():
            test_samples.append(json.loads(_path.read_text()))
    test = raw_samples_to_dataset(test_samples)
    test_dataloader = DataLoader(
            test.map(preprocessor).save(cachedir / "test.cache"),
            sampler=SequentialSampler(test),
            batch_size=batch_size,
            num_workers=8,
            )

    return train_dataloader, val_dataloader, test_dataloader


class DataModule(L.LightningDataModule):
    def __init__(self, datadir, batch_size):
        super().__init__()
        train_dataloader, val_dataloader, test_dataloader = get_dataloader(datadir, batch_size)
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._test_dataloader = test_dataloader
    def train_dataloader(self):
        return self._train_dataloader
    def val_dataloader(self):
        return self._val_dataloader
    def test_dataloader(self):
        return self._test_dataloader

class Pipeline(L.LightningModule):

    def __init__(self):
        super().__init__()
        model = BertForMultipleChoice.from_pretrained("bert-base-uncased", num_labels=NUM_LABELS)
        self.model = model
        self.outputs = []

    def training_step(self, batch, batch_idx):
        labels = batch["label"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        model_outs = self.model(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels
                )
        loss = model_outs.loss
        logits = model_outs.logits
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        model_outs = self.model(
                input_ids,
                token_type_ids=token_type_ids,
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
        self.log('val_loss', val_loss)
        self.log('val_acc', val_acc)        
        self.outputs.clear()
        return val_loss

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        weight_decay = 0.0
        adam_epsilon = 1e-8

        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay
                },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                }
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=adam_epsilon)

        return optimizer

if __name__ == "__main__":
    L.seed_everything(42)
    config = {
        "batch_size": 24,
        }

    early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.0,
            patience=2,
            verbose=True,
            mode="min",
            )

    checkpoint_filename = 'bert-base-uncased'
    logger = CSVLogger(save_dir="logs", name=f'race', version=checkpoint_filename)
    dm = DataModule(datadir=f'./datasets/RACE', batch_size=config["batch_size"])
    pipeline = Pipeline()
    trainer = L.Trainer(
            accelerator="gpu", devices=[0,1],
            max_epochs=10,
            logger=logger,
            callbacks=[early_stop_callback],
            )
    trainer.fit(pipeline, dm)
    trainer.test(pipeline, dm)
