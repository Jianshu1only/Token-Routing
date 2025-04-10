# 🔀 Token-Level Routing for Edge Inference

> Efficient collaborative decoding between edge-deployed small models and cloud-based large language models.

## 🎬 Demo

![System Overview](Front_end.png)

🎥 **Watch the demo**:  
[![Watch the demo](https://img.youtube.com/vi/Tr_ziV_PJT4/hqdefault.jpg)](https://www.youtube.com/watch?v=Tr_ziV_PJT4)

---

## 🧠 Overview

This project implements **Token-Level Routing**, a novel **collaborative inference system** where a small on-device model performs most decoding, selectively routing **critical tokens** to a powerful **cloud-based LLM**.

This approach significantly reduces latency and cost while retaining output quality — ideal for **edge scenarios** like mobile phones or IoT devices.

---

## 🚀 Key Features

- ⚡ **Efficient**: >60% accuracy boost by routing only ~7% of tokens to the LLM.
- 🌐 **Edge-Cloud Collaboration**: Combines local lightweight models with cloud intelligence.
- 🧭 **Token-Level Routing**: Fine-grained, confidence-driven token control.
- 📱 **Deployable**: Lightweight ONNX runtime works on laptops and mobile devices.
- 🖥️ **LLM Backend**: Compatible with [SGLang] for LLM serving and kv-cache extension.

---

## 🧩 Architecture


```text
+-------------+           +-------------+           +-------------+
|  User Input |--Prompt-->|  SLM (ONNX) |--Tokens-->|   Router     |
+-------------+           +-------------+           +-------------+
                                                 |
                            Tokens with low confidence
                                                 v
                                      +------------------+
                                      | LLM (Server-side)|
                                      +------------------+
```
---
## 📘 Usage

See [**Guideline.md**](./Guideline.md) for setup and usage instructions.

---

## 💻 Platform Support

- ✅ **macOS (Apple M1/M2/M3) are already support**
- 🔧 **Android under development！**

---

## 📫 Contact

For questions or collaborations, feel free to open an issue or email us.
