import torch
from src.utils import get_torch_data
from src.utils import model_from_hf
from transformers import BertTokenizer


def export_model_to_onnx(torch_model_name: str, onnx_model_name: str, input_example):
    merged_model = torch.load(f"models/torch/{torch_model_name}.pt", weights_only=False, map_location="cpu")
    merged_model.eval()
    
    torch.onnx.export(
        model=merged_model,
        args=(input_example["input_ids"].unsqueeze(0), input_example["attention_mask"].unsqueeze(0)),
        f=f"models/onnx/{onnx_model_name}.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len",},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size"}
        },
        verbose=True,
        dynamo=False,
        do_constant_folding=False,
    )

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(model_from_hf)
    
    data = get_torch_data("data/raw.csv", tokenizer=tokenizer)
    
    
    torch_model_name = "merged_best_model"
    onnx_model_name = "prod_model_single_sample"
    
    export_model_to_onnx(torch_model_name, onnx_model_name, data["train"][0])