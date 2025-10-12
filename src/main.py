from datasets import load_dataset, DatasetDict, Dataset
from peft import LoraConfig, LoraModel, get_peft_model
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader
import pandas as pd
from typing import Collection

import transformers
import torch
import torch.nn as nn

author_to_label = {
    "blok": 0,
    "cvetaeva": 1,
    "pasternak": 2
}
transformers.logging.set_verbosity_error()

def preprocess_data(data: pd.DataFrame):
    data["text_length"] = data["text"].str.split().str.len()
    data = data[data["text_length"] < 1000]
    data["label"] = data["label"].apply(lambda x: author_to_label[x])
    return data

def df_to_dataset(df: pd.DataFrame) -> DatasetDict:
    data = Dataset.from_pandas(df)
    data = data.shuffle()
    data = data.train_test_split(test_size=0.2)
    return data

def tokenize_text(data: Dataset, tokenizer: BertTokenizer):
    return tokenizer(
        data["text"],
        padding=True,
        truncation=False
    )


model_name = "cointegrated/rubert-tiny2"

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
tokenizer = BertTokenizer.from_pretrained(model_name)


data = pd.read_csv("dataset/output.csv")
data = preprocess_data(data)
dataset = df_to_dataset(data)
dataset = dataset.map(lambda x: tokenize_text(x, tokenizer), batched=True, batch_size=64)

train_data_loader = DataLoader(dataset["train"], batch_size=16, shuffle=False)
test_data_loader = DataLoader(dataset["test"], batch_size=16, shuffle=False)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query", "value", "key", "output.dense"],
    lora_dropout=0.05,
    task_type='SEQ_CLS'
)

lora_model = get_peft_model(model, lora_config)
optimizer = torch.optim.AdamW(lora_model.parameters(), lr=5e-5)

exit(0)

for epoch in range(3):
    total_train_loss = 0
    total_test_loss = 0
    
    lora_model.train()
    for batch in train_data_loader:
        optimizer.zero_grad()
        outputs = lora_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["label"]
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    
    lora_model.eval()
    for batch in test_data_loader:
        pass