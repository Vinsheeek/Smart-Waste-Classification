import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ✅ Class labels (VERY IMPORTANT)
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# ✅ Same transforms as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ✅ Load model
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, 6)

model.load_state_dict(torch.load("waste_model.pth"))
model.eval()

# ✅ Load test image
img = Image.open("Garbage classification Dataset/glass/glass19.jpg")  
img = transform(img).unsqueeze(0)

# ✅ Prediction
output = model(img)

probabilities = torch.nn.functional.softmax(output[0], dim=0)
confidence = torch.max(probabilities)

_, pred = torch.max(output, 1)

print("Prediction:", classes[pred.item()])
print("Confidence:", round(confidence.item() * 100, 2), "%")