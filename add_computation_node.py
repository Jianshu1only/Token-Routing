import onnx
from onnx import helper, TensorProto

model_path = "qwen2.5-0.5b.onnx"
model = onnx.load(model_path)

original_output_name = "/model/model/layers.23/post_attention_layernorm/Mul_1_output_0"
target_output_name = "/model/model/layers.23/post_attention_layernorm/Mul_1_hidden_output"

candidate_name = None
for node in model.graph.node:
    for out in node.output:
        if out == original_output_name:
            candidate_name = out
            break
    if candidate_name is not None:
        break

if candidate_name is None:
    raise ValueError(f"can not find '{original_output_name}' node, please confirm node name。")
else:
    print(f"Find node: {candidate_name}")


identity_node = helper.make_node(
    "Identity",
    inputs=[candidate_name],
    outputs=[target_output_name],
    name="Identity_post_attention_layernorm_Mul_1_hidden"
)

model.graph.node.append(identity_node)

new_output = helper.make_tensor_value_info(
    target_output_name, 
    TensorProto.FLOAT16, 
    [None, None, 896]
)
model.graph.output.append(new_output)

modified_model_path = "qwen2.5-0.5b_with_hidden.onnx"
onnx.save(model, modified_model_path)
print(f"Save modified model to {modified_model_path}")
