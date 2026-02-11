import torch
from transformers.models import BertForSequenceClassification
from peft import get_peft_model
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from copy import deepcopy

class CustomTrainer:
    def __init__(self, train_data_loader: DataLoader, test_data_loader: DataLoader, delta=0.002, device="cpu"):
        self.device: str = device
        
        self.train_data_loader: DataLoader = train_data_loader
        self.test_data_loader: DataLoader = test_data_loader
        
        self.delta = delta
        
        self.lora_model = None

    def get_basic_model(self, model_name):
        self.clean_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    def set_training_model(self, lora_config):
        model = deepcopy(self.clean_model).to(self.device)
        self.lora_model = get_peft_model(model, lora_config)
        

    def training_loop(self, lr, num_epochs=100, verbose=False, save_models=False, early_stop=True, model_name="run"):
        if self.lora_model is None:
            print("You should initiate lora_model first")
            return -2
        
        optimizer = torch.optim.AdamW(self.lora_model.parameters(), lr=lr)
        patience_counter, best_epoch, best_f1 = 0, -1, -1
        
        for epoch in range(num_epochs):
            
            total_train_loss, total_test_loss = 0, 0
            self.lora_model.train()
            for batch in self.train_data_loader:
                optimizer.zero_grad()
                outputs = self.lora_model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    labels=batch["label"].to(self.device)
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                
            total_train_loss /= len(self.train_data_loader)
            
            self.lora_model.eval()
            ys, ps = [], []
            with torch.no_grad():
                for batch in self.test_data_loader:
                    test_outputs = self.lora_model(
                        input_ids=batch["input_ids"].to(self.device),
                        attention_mask=batch["attention_mask"].to(self.device),
                        labels=batch["label"].to(self.device)
                    )
                    total_test_loss += test_outputs.loss.item()
                    predictions = torch.argmax(test_outputs.logits, dim=1)
                    ys.append(batch["label"].cpu())
                    ps.append(predictions.detach().cpu())
            y = torch.concatenate(ys)
            p = torch.concatenate(ps)
            
            f1 = f1_score(y, p, average="macro")
            total_test_loss /= len(self.test_data_loader)
            
            if f1 > best_f1 + self.delta:
                best_state = {k: v.detach().cpu() for k, v in self.lora_model.merge_and_unload().state_dict().items()}
                patience_counter = 0
                best_f1 = f1
                best_epoch = epoch
            else:
                patience_counter += 1
                if patience_counter == 5 and early_stop:
                    print("Early stopping...")
                    break
            if verbose:
                print(f"Epoch #{epoch + 1}")
                print(f'Avg train loss: {total_train_loss}')
                print(f'Avg test loss: {total_test_loss}')
                print(f"Macro f1: {f1}")
                print(f"-" * 30)
                
        if save_models:
            torch.save(best_state, f"/models/torch/{model_name}.pt")
        return best_f1, best_epoch