import os
import sys
import time
import numpy as np
import torch
import onnxruntime
import requests
import onnx
from transformers import AutoTokenizer, AutoConfig
import streamlit as st
from token_conf_model import Qwen2Confidence
from onnx import external_data_helper
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# === Config ===
onnx_small_model = os.environ.get('ONNX_SMALL_MODEL', './qwen0.5_trained/qwen2.5-0.5b_with_hidden.onnx')
tokenizer_name = os.environ.get('TOKENIZER', 'Qwen/Qwen2.5-32B-Instruct')
mlp_model_path = os.environ.get('MLP_MODEL', './mlp/mlp_iter_3407.pth')

if not all([onnx_small_model, tokenizer_name, mlp_model_path]):
    st.error("Missing required model path or tokenizer configuration.")
    sys.exit(1)

class Args:
    def __init__(self):
        self.onnx_small_model = onnx_small_model
        self.tokenizer = tokenizer_name
        self.mlp_model = mlp_model_path
        self.threshold = 0.4
        self.max_tokens = 50
        self.inference_mode = 'joint'
        self.external_folder = None

args = Args()

class ONNXInferenceSystem:
    def __init__(self, model_path, tokenizer_name, external_folder=None):
        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ("MetalExecutionProvider" in onnxruntime.get_all_providers())
        providers = ["MetalExecutionProvider", "CPUExecutionProvider"] if providers else ["CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(model_path, sess_options=options, providers=providers)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate_next_token(self, input_ids: np.ndarray):
        onnx_inputs = {self.session.get_inputs()[0].name: input_ids}
        logits, hidden_states = self.session.run(None, onnx_inputs)
        next_token_logits = logits[:, -1, :]
        last_hidden = hidden_states[:, -1, :]
        next_token_ids = np.argmax(next_token_logits, axis=-1)[:, np.newaxis]
        return next_token_ids, last_hidden

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
onnx_system = ONNXInferenceSystem(args.onnx_small_model, args.tokenizer, args.external_folder)

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
tokenizer.padding_side = "left"
tokenizer.pad_token_id = tokenizer.eos_token_id

mlp_config = AutoConfig.from_pretrained(args.tokenizer)
mlp_config.hidden_size = 896
mlp_model = Qwen2Confidence(mlp_config).to(device)
state_dict = torch.load(args.mlp_model, map_location=device)
mlp_model.load_state_dict(state_dict)
mlp_model.eval()

def request_large_model_token(prompt_text, tokenizer):
    port = 30000
    url = "https://11da-5-195-0-145.ngrok-free.app/v1/completions"
    #url = f"http://localhost:{port}/v1/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "Qwen/Qwen2.5-32B-Instruct",
        "prompt": prompt_text,
        "max_new_tokens": 2,
        "temperature": 0.0,
        "stop": ["\n", ".", ",", "!", "?"]
    }
    try:
        response = requests.post(url, headers=headers, json=data, verify=False)
        result = response.json()
        generated_text = result["choices"][0]["text"]
        if not prompt_text.endswith(' ') and not generated_text.startswith(' '):
            generated_text = ' ' + generated_text
        token_ids = tokenizer.encode(generated_text, add_special_tokens=False)
        return token_ids[0] if token_ids else tokenizer.eos_token_id
    except Exception:
        return tokenizer.eos_token_id

def run_inference(prompt, threshold, max_tokens, inference_mode, stream=True):
    encoded = tokenizer(prompt, return_tensors="np")
    input_ids_np = encoded["input_ids"]
    prompt_len = int(encoded["attention_mask"].sum())
    sequence = list(input_ids_np[0][:prompt_len])
    
    # Profiling 变量初始化
    start_inference_time = time.perf_counter()
    time_to_first_token = None
    total_small_inference_time = 0.0  
    total_large_inference_time = 0.0  
    total_comm_time = 0.0           
    routed_tokens = 0
    history = []

    while total_tokens < max_tokens:
        token_start_time = time.perf_counter()
        input_batch = np.array([sequence])
        
        start_small = time.perf_counter()
        next_tokens, last_hidden = onnx_system.generate_next_token(input_batch)
        small_inference_time = time.perf_counter() - start_small
        total_small_inference_time += small_inference_time

        mlp_input = torch.from_numpy(last_hidden[0]).unsqueeze(0).unsqueeze(1).to(device).float()
        start_mlp = time.perf_counter()
        decision_score = mlp_model(mlp_input).item()
        mlp_inference_time = time.perf_counter() - start_mlp
        total_small_inference_time += mlp_inference_time

        if total_tokens == 0:
            time_to_first_token = time.perf_counter() - start_inference_time

        if inference_mode == "joint" and decision_score < threshold:
            full_prompt = tokenizer.decode(sequence, skip_special_tokens=True)
            start_large = time.perf_counter()
            comm_start = time.perf_counter()
            new_token = request_large_model_token(full_prompt, tokenizer)
            comm_time = time.perf_counter() - comm_start
            large_inference_time = time.perf_counter() - start_large

            total_large_inference_time += large_inference_time
            total_comm_time += comm_time
            source = "large"
            routed_tokens += 1
        else:
            new_token = int(next_tokens[0, 0])
            source = "small"

        sequence.append(new_token)
        total_tokens += 1
        token_str = tokenizer.decode([new_token], skip_special_tokens=True)
        history.append((token_str, source))
        
        if stream:
            if source == "large":
                yield f"<span style='color: red;'>{token_str}</span>"
            else:
                yield token_str

        if new_token == tokenizer.eos_token_id:
            break

    total_duration = time.perf_counter() - start_inference_time
    avg_token_time = total_duration * 1000 / total_tokens if total_tokens > 0 else 0

    profiling_info = {
        "Time to first token (ms)": time_to_first_token * 1000 if time_to_first_token else None,
        "Total tokens": total_tokens,
        "Routed tokens": routed_tokens,
        "Percentage routed (%)": (routed_tokens / total_tokens * 100) if total_tokens > 0 else 0,
        "Total small model inference time (ms)": total_small_inference_time * 1000,
        "Total large model inference time (ms)": total_large_inference_time * 1000,
        "Total communication time (ms)": total_comm_time * 1000,
        "Average time per token (ms)": avg_token_time,
        "Total inference duration (ms)": total_duration * 1000,
    }
    profiling_output = "<br/><br/><b>Profiling Info:</b><br/>"
    for key, value in profiling_info.items():
        if isinstance(value, float):
            profiling_output += f"{key}: {value:.2f}<br/>"
        else:
            profiling_output += f"{key}: {value}<br/>"

    if stream:
        yield profiling_output
    else:
        final_output = "".join(
            f"<span style='color: red;'>{tok}</span>" if src == "large" else tok
            for tok, src in history
        )
        final_output += profiling_output
        yield final_output

st.title("Stream Inference Comparison")

prompt_input = st.text_input("Enter prompt", value="", placeholder="e.g., Hello, world!", key="prompt", help="Prompt to be used for generation")

col1, col2, col3, col4 = st.columns([1.2, 1, 1.3, 1])

with col1:
    threshold = st.number_input("Threshold", min_value=0.0, max_value=1.0, value=args.threshold, step=0.01)

with col2:
    max_tokens = st.number_input("Max tokens", min_value=1, value=args.max_tokens)

with col3:
    inference_mode = st.selectbox("Mode", ["small_only", "joint"], index=["small_only", "joint"].index(args.inference_mode))

with col4:
    stream_mode = st.checkbox("Streaming", value=True)

if st.button("Start Inference") and prompt_input:
    output_area = st.empty()
    generated_text = ""
    adjusted_threshold = threshold / 2
    for chunk in run_inference(prompt_input, adjusted_threshold, max_tokens, inference_mode, stream=stream_mode):
        generated_text += chunk if stream_mode else chunk
        output_area.markdown(generated_text, unsafe_allow_html=True)

st.write("Inference finished")
