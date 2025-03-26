import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class QwenONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        outputs = self.model(input_ids, use_cache=False)
        return outputs[0]

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model.eval()

wrapper = QwenONNXWrapper(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wrapper.to(device)

prompt = "Hi, how are you today?"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)

output_path = "qwen2.5-0.5b.onnx"
torch.onnx.export(
    wrapper,                          
    input_ids,                        
    output_path,                      
    input_names=["input_ids"],       
    output_names=["logits"],       
    dynamic_axes={                    
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"}
    },
    opset_version=14,                 
)

print(f"ONNX has been exported to {output_path}")
