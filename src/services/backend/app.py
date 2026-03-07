from fastapi import FastAPI
from pydantic import BaseModel
from src.services.backend.model import model
from utils import model_from_hf, label_to_author
from transformers import AutoTokenizer
from scipy.special import softmax

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained(model_from_hf)

class Process(BaseModel):
    text: str

@app.post("/process")
def process(input: Process):
    tokenized_text = tokenizer(
        input.text, 
        return_tensors="pt"
    )
    logits = model.invoke(tokenized_text)[0]
    logits = [x.tolist() for x in logits]
    logits = softmax(logits, axis=1)
    res = []
    for x in logits:
        x = {label_to_author[idx]: v for idx, v in enumerate(x)}
        res.append(x)
    return {"logits": res}