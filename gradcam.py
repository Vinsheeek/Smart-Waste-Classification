import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms, models
from PIL import Image

classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ✅ Load Model
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, 6)
model.load_state_dict(torch.load("waste_model.pth"))
model.eval()

# ✅ Load Image
img_path = "Garbage classification Dataset/plastic/plastic139.jpg"
image = Image.open(img_path).convert('RGB')

input_tensor = transform(image).unsqueeze(0)

# ✅ Hook for Gradients
gradients = []
activations = []

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

def forward_hook(module, input, output):
    activations.append(output)

# ✅ Target Layer (VERY IMPORTANT)
target_layer = model.features[-1]

target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# ✅ Forward Pass
output = model(input_tensor)
pred_class = torch.argmax(output)

# ✅ Backward Pass
model.zero_grad()
output[0, pred_class].backward()

# ✅ Grad-CAM Logic
grads = gradients[0]
acts = activations[0]

weights = torch.mean(grads, dim=(2, 3))

cam = torch.zeros(acts.shape[2:], dtype=torch.float32)

for i, w in enumerate(weights[0]):
    cam += w * acts[0, i]

cam = np.maximum(cam.detach().numpy(), 0)
cam = cam / cam.max()
cam = cv2.resize(cam, (224, 224))

# ✅ Convert Image
image_np = np.array(image.resize((224, 224)))

heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

# ✅ Display
plt.imshow(overlay)
plt.axis('off')
plt.title(f"Grad-CAM → {classes[pred_class]}")
plt.show()