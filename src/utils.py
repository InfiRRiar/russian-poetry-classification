import pandas as pd
from datasets import DatasetDict, Dataset, ClassLabel
from transformers import BertTokenizer
import torch

author_to_label = {
    "blok": 0,
    "cvetaeva": 1,
    "pasternak": 2
}

model_from_hf = "cointegrated/rubert-tiny2"


def preprocess_data(data: pd.DataFrame):
    processed_data = data.copy()
    processed_data["text_length"] = processed_data["text"].str.split().str.len()
    processed_data["label"] = processed_data["label"].apply(lambda x: author_to_label[x])
    return processed_data

def df_to_dataset(df: pd.DataFrame) -> DatasetDict:
    data = Dataset.from_pandas(df)
    data = data.cast_column("label", ClassLabel(num_classes=3))
    data = data.shuffle()
    data = data.train_test_split(test_size=0.2, stratify_by_column="label")
    return data

def tokenize_text(data: Dataset, tokenizer: BertTokenizer):
    return tokenizer(
        data["text"],
        padding="max_length",
        truncation=True,
        max_length=500
    )
    
def get_torch_data(path: str, tokenizer: BertTokenizer):
    data = pd.read_csv(path)
    data = preprocess_data(data)
    dataset = df_to_dataset(data)
    dataset = dataset.map(lambda x: tokenize_text(x, tokenizer), batched=True, batch_size=64)
    dataset.set_format(type="torch")
    
    return dataset