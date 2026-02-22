import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ✅ Device config (important for interviews)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ✅ Image preprocessing + augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ✅ Load dataset
dataset = datasets.ImageFolder("Garbage Classification Dataset/", transform=transform)

# ✅ Train / Validation Split (VERY IMPORTANT)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ✅ Load pretrained MobileNetV2
model = models.mobilenet_v2(pretrained=True)

# ✅ Modify classifier layer
model.classifier[1] = nn.Linear(model.last_channel, 6)

model = model.to(device)

# ✅ Loss + Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ Training Loop
epochs = 5

for epoch in range(epochs):

    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # ✅ Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(f"Epoch [{epoch+1}/{epochs}]")
    print(f"Loss: {running_loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.2f}%")
    print("-" * 30)

# ✅ Save Model
torch.save(model.state_dict(), "waste_model.pth")

print("✅ Training Complete. Model Saved.")