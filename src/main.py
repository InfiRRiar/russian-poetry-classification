from datasets import load_dataset, DatasetDict, Dataset
from peft import LoraConfig, LoraModel, get_peft_model
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader
import pandas as pd
from typing import Collection
from tqdm import tqdm

import transformers
import torch
import torch.nn as nn

author_to_label = {
    "blok": 0,
    "cvetaeva": 1,
    "pasternak": 2
}
transformers.logging.set_verbosity_error()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

def preprocess_data(data: pd.DataFrame):
    processed_data = data.copy()
    processed_data["text_length"] = processed_data["text"].str.split().str.len()
    processed_data = processed_data[processed_data["text_length"] < 1000]
    processed_data["label"] = processed_data["label"].apply(lambda x: author_to_label[x])
    return processed_data

def df_to_dataset(df: pd.DataFrame) -> DatasetDict:
    data = Dataset.from_pandas(df)
    data = data.shuffle()
    data = data.train_test_split(test_size=0.2)
    return data

def tokenize_text(data: Dataset, tokenizer: BertTokenizer):
    return tokenizer(
        data["text"],
        padding=True,
        truncation=False,
    )


model_name = "cointegrated/rubert-tiny2"

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)
tokenizer = BertTokenizer.from_pretrained(model_name)


data = pd.read_csv("dataset/output.csv")
data = preprocess_data(data)
dataset = df_to_dataset(data)
dataset = dataset.map(lambda x: tokenize_text(x, tokenizer), batched=True, batch_size=64)
dataset.set_format(type="torch")

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

for epoch in range(3):
    print(f"Epoch #{epoch + 1}")
    print(f"-" * 30)
    total_train_loss = 0
    total_test_loss = 0
    
    lora_model.train()
    for batch in tqdm(train_data_loader):
        optimizer.zero_grad()
        outputs = lora_model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["label"].to(device)
        )
        loss = outputs.loss
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    total_train_loss /= len(train_data_loader)
    print(f"Total train loss: {total_train_loss}")
    
    lora_model.eval()
    for batch in test_data_loader:
        test_outputs = lora_model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["label"].to(device)
        )
        total_test_loss += test_outputs.loss.item()
        
    total_test_loss /= len(test_data_loader)
    print(f"Total test loss: {total_test_loss}")
    print(f"-" * 30)
    
torch.save(model.state_dict(), "models/alpha.pt")