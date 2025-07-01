import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# Modello aggiornato: non carica pesi preaddestrati e può essere usato con dataset non etichettato
class SegmentationAwareClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(SegmentationAwareClassifier, self).__init__()
        self.backbone = models.resnet18(weights=None)  # Nessun peso preaddestrato
        self.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def train_model(model, train_loader: DataLoader, val_loader: DataLoader, device, epochs=10, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, segms, labels in train_loader:
            imgs, segms, labels = imgs.to(device), segms.to(device), labels.to(device)
            if segms.ndim == 3:
                segms = segms.unsqueeze(1)
            inputs = torch.cat((imgs, segms), dim=1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        val_loss, val_acc = evaluate_model(model, val_loader, device)

        print(f"Epoch {epoch+1}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

    if best_model_state:
        model.load_state_dict(best_model_state)

    return model


def evaluate_model(model, loader: DataLoader, device):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for imgs, segms, labels in loader:
            imgs, segms, labels = imgs.to(device), segms.to(device), labels.to(device)
            if segms.ndim == 3:
                segms = segms.unsqueeze(1)
            inputs = torch.cat((imgs, segms), dim=1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    return total_loss / len(loader), acc


