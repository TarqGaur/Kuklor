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




import cv2
import time
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access webcam.")
    exit()

frame_interval_seconds = 1
frame_idx = 0

print("Starting webcam capture and captioning with BLIP...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Convert OpenCV BGR frame to PIL RGB image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        try:
            # Generate caption (unconditional)
            inputs = processor(pil_image, return_tensors="pt")
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)

            print(f"Frame {frame_idx}: {caption}")

            # Save caption
            with open("captions_blip.txt", "a") as f:
                f.write(f"Frame {frame_idx}: {caption}\n")

        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")

        frame_idx += 1
        time.sleep(frame_interval_seconds)  # Wait 1 second

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released.")
