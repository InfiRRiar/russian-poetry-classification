import mlflow
import numpy as np
import pandas as pd
import random
import transformers
import torch
import optuna
import os

from datasets import DatasetDict, Dataset, ClassLabel
from peft import LoraConfig
from transformers import BertTokenizer
from torch.utils.data import DataLoader

from src.sft_process.custom_trainer import CustomTrainer

from src.utils import get_torch_data
from src.utils import model_from_hf


device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = BertTokenizer.from_pretrained(model_from_hf)

def set_seed(seed: int):
    torch.manual_seed(seed=seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


EXP_NAME = "MLflow engage #1"

mlflow.set_experiment(EXP_NAME)


def get_optuna_hyperparameters(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    r = trial.suggest_categorical("r", [8, 16, 32, 64, 128])
    multiplicator = trial.suggest_int("alpha_multiplicator", 1, 4)
    lora_dropout = trial.suggest_float("lora_dropout", 0, 0.15)
    
    lora_config = LoraConfig(
        r=r,
        lora_alpha=r*multiplicator,
        target_modules=["query", "value", "key", "output.dense"],
        lora_dropout=lora_dropout,
        task_type='SEQ_CLS'
    )
    
    return lora_config, lr

def optuna_search(trial, trainer: CustomTrainer):
    with mlflow.start_run(run_name=f"optuna-trial-{trial.number}"):
        lora_config, lr = get_optuna_hyperparameters(trial)
        mlflow.log_params(lora_config.to_dict())
        
        trainer.set_training_model(lora_config)
        f1, epoch = trainer.training_loop(lr, 150)
        mlflow.log_metric("f1", f1)
        mlflow.log_param("best_epoch", epoch)
        mlflow.log_param("lr", lr)
        
        print(f"Trial finished. f1: {f1}")
        return f1

def train_best_model(trainer: CustomTrainer, model_name: str):
    set_seed(42)
    
    exp = mlflow.get_experiment_by_name(EXP_NAME)
    best_run = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.f1 DESC"]
    ).iloc[0]

    best_config = best_run.filter(like="params.").to_dict()
    best_config = {k[len("params."):]: v for k, v in best_config.items() if k.startswith("params.")}

    best_epoch_counter = int(best_config.pop("best_epoch")) + 1
    best_lr = float(best_config.pop("lr"))

    best_lora_config = LoraConfig(
        r=int(best_config["r"]),
        lora_alpha=int(best_config["lora_alpha"]),
        target_modules=["query", "value", "key", "output.dense"],
        lora_dropout=float(best_config["lora_dropout"]),
        task_type="SEQ_CLS"
    )
    trainer.set_training_model(best_lora_config)
    trainer.training_loop(
        lr=best_lr,
        num_epochs=best_epoch_counter,
        verbose=True,
        save_models=True,
        early_stop=False,
        model_name=model_name
    )

def main():
    dataset = get_torch_data("data/raw.csv", tokenizer=tokenizer)

    train_data_loader = DataLoader(dataset["train"], batch_size=16, shuffle=True, pin_memory=True)
    test_data_loader = DataLoader(dataset["test"], batch_size=16, shuffle=True, pin_memory=True,)

    trainer = CustomTrainer(
        train_data_loader=train_data_loader, 
        test_data_loader=test_data_loader, 
        device=device
    )
        
    print(dataset["train"][0]["input_ids"])
    
    trainer.get_basic_model(model_from_hf)
    
    study = optuna.create_study(study_name="study", direction="maximize")
    study.optimize(lambda trial: optuna_search(trial=trial, trainer=trainer), n_trials=2)


    model_name = "optuna_best_model"

    train_best_model(trainer, model_name)

if __name__ == "__main__":
    main()