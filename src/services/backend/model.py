import onnxruntime as ort
import numpy as np

class Model:
    def __init__(self):
        self.session = ort.InferenceSession(
            path_or_bytes="models/onnx/prod_model_single_sample.onnx",
            providers=["CUDAExecutionProvider"]
        )
        
    def invoke(self, input_data):
        input_data = {
            "input_ids": input_data["input_ids"],
            "attention_mask": input_data["attention_mask"] 
        }
        input_data = {k: v.detach().cpu().numpy().astype(np.int64) for k, v in input_data.items()}
        outputs = self.session.run(["logits"], input_data)
        return outputs

model = Model()