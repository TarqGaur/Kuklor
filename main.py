# from transformers import pipeline
# pipe = pipeline("text-classification")
# res = pipe(["This restaurant is awesome", "This restaurant is awful"])
# print(res)





#This is for Image
# import warnings
# warnings.filterwarnings("ignore")

# from transformers import pipeline

# captioner = pipeline(model="ydshieh/vit-gpt2-coco-en")
# res = captioner("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
# print(res)



#THis s for Video
# from transformers import pipeline
# classifier = pipeline(task="video-classification", model="MCG-NJU/videomae-base-finetuned-kinetics")
# result = classifier("vidt.mp4", top_k=5)
# print(result)


# import google.generativeai as genai
# import PIL.Image

# genai.configure(api_key="AIzaSyAmei6RqPdLYC6tyHWllajLSNhqjUplkCw")

# # Load your image
# image = PIL.Image.open("parrots.png")  # Example: "cat.jpg"

# # Initialize the model
# model = genai.GenerativeModel('gemini-1.5-flash')

# # Make the caption request
# response = model.generate_content(
#     [image, "Describe this image in one short sentence."]
# )

# # Print the caption
# print(response.text)



# from transformers import AutoProcessor, AutoModelForCausalLM
# import torch
# from PIL import Image

# # Load model and processor
# model_id = "OpenGVLab/InternVL3-1B"
# model = AutoModelForCausalLM.from_pretrained(model_id)
# processor = AutoProcessor.from_pretrained(model_id)

# # Prepare image
# image = Image.open("parrots.png.jpg")

# # For image captioning
# prompt = "Describe this image:"
# inputs = processor(text=prompt, images=image, return_tensors="pt")

# # Generate caption
# with torch.no_grad():
#     output = model.generate(
#         **inputs,
#         max_new_tokens=100,
#         do_sample=False
#     )

# # Decode the output
# generated_text = processor.decode(output[0], skip_special_tokens=True)
# print(generated_text)




# Use a pipeline as a high-level helper
# from PIL import Image
# from transformers import pipeline

# img = Image.open("parrots.png")
# classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")
# classifier(img)




# salesforce 

import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

img_url = 'http://192.168.1.14:8080/shot.jpg'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# conditional image captioning
text = "a photography of"
inputs = processor(raw_image, text, return_tensors="pt")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
# >>> a photography of a woman and her dog

# unconditional image captioning
inputs = processor(raw_image, return_tensors="pt")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
