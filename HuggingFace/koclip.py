import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

repo = "Bingsu/clip-vit-large-patch14-ko"
model = AutoModel.from_pretrained(repo)
processor = AutoProcessor.from_pretrained(repo)

## 이미지 경로 지정
image_path = 'dog.png'

image = Image.open(image_path).convert('RGB')

## 텍스트 쿼리 지정
text_query = ["고양이 사진", "강아지 사진"]
inputs = processor(text=text_query, images=image, return_tensors="pt", padding=True)

with torch.inference_mode():
    outputs = model(**inputs)

logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

pred_idx = torch.argmax(probs[0])
pred_text = text_query[pred_idx]

print(pred_text)
