from PIL import Image
import torch

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

## 이미지 경로 지정
image_path = 'dog.png'

image = Image.open(image_path).convert('RGB')

## 텍스트 쿼리 지정
text_query = ["a photo of a cat", "a photo of a dog"]

inputs = processor(text=text_query, images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  
probs = logits_per_image.softmax(dim=1) 

pred_idx = torch.argmax(probs[0]).item()
pred_text = text_query[pred_idx]

print(f'pred : {pred_text}')
