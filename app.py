import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
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

st.title("Smart Waste Classification ♻️")

uploaded_file = st.file_uploader("Upload Waste Image")

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image")

    input_tensor = transform(image).unsqueeze(0)

    # ✅ Hooks for Grad-CAM
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.features[-1]

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    # ✅ Forward Pass
    output = model(input_tensor)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    confidence = torch.max(probabilities)

    _, pred = torch.max(output, 1)

    st.write("### Prediction:", classes[pred.item()])
    st.write("Confidence:", round(confidence.item() * 100, 2), "%")

    # ✅ Backward Pass for Grad-CAM
    model.zero_grad()
    output[0, pred].backward()

    grads = gradients[0]
    acts = activations[0]

    weights = torch.mean(grads, dim=(2, 3))

    cam = torch.zeros(acts.shape[2:], dtype=torch.float32)

    for i, w in enumerate(weights[0]):
        cam += w * acts[0, i]

    cam = np.maximum(cam.detach().numpy(), 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (224, 224))

    image_np = np.array(image.resize((224, 224)))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

    st.image(overlay, caption="Grad-CAM Visualization 🔥")