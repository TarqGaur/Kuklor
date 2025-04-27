from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from PIL import Image
import requests
from io import BytesIO

# Download the image
url = "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg"
image = Image.open(BytesIO(requests.get(url).content))

# Load model and processor
processor = AutoProcessor.from_pretrained("OpenGVLab/InternVL3-1B")
model = AutoModelForCausalLM.from_pretrained("OpenGVLab/InternVL3-1B", torch_dtype=torch.float16, device_map="auto")

# Process the image and generate text
prompt = "Describe this image in detail."
inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

# Generate
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )

# Print output
generated_text = processor.decode(output[0], skip_special_tokens=True)
print(generated_text)
