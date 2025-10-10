from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset
import torch
import torch.nn as nn


model_name = "cointegrated/rubert-tiny2"

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
tokenizer = BertTokenizer.from_pretrained(model_name)

data = ["Some stuff"]

data = tokenizer(data, return_tensors="pt")
outputs = model(data["input_ids"])

loss = nn.CrossEntropyLoss()
