import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16  # MPS genelde fp16 ile daha sorunsuz

processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device)

# image = Image.open("test.jpg")
# prompt formatı OpenVLA README’deki gibi
prompt = "In: What action should the robot take to {move to the red object}?\n Out:"

# inputs = processor(prompt, image, return_tensors="pt").to(device)
# action = vla.predict_action(**inputs, unnorm_key=..., do_sample=False)