import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import DATA_DIR, MODEL_NAME, NUM_CLASSES, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, IMG_SIZE, DEVICE, SAVE_DIR
from dataset import create_dataloaders
from model import get_model, count_parameters


def train():
    device = DEVICE if torch.cuda.is_available() else 'cpu'
    print(f"Устройство: {device}")

    train_loader, val_loader, class_names = create_dataloaders(DATA_DIR, BATCH_SIZE, IMG_SIZE)

    model = get_model(MODEL_NAME, num_classes=len(class_names), pretrained=True)
    model = model.to(device)
    print(f"Модель: {MODEL_NAME}, параметров: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(NUM_EPOCHS):
        # --- train ---
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
        train_loss /= len(train_loader)
        train_acc = 100.0 * correct / total

        # --- val ---
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                correct += outputs.argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        val_loss /= len(val_loader)
        val_acc = 100.0 * correct / total

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        marker = " [BEST]" if val_acc > best_val_acc else ""
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.1f}% | "
              f"val_loss={val_loss:.4f} acc={val_acc:.1f}%{marker}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {'model_name': MODEL_NAME, 'num_classes': len(class_names), 'img_size': IMG_SIZE},
                'class_names': class_names,
                'best_val_acc': best_val_acc,
            }, SAVE_DIR / 'best_model.pth')

    torch.save({'model_state_dict': model.state_dict(),
                'config': {'model_name': MODEL_NAME, 'num_classes': len(class_names), 'img_size': IMG_SIZE},
                'class_names': class_names}, SAVE_DIR / 'last_model.pth')

    with open(SAVE_DIR / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nГотово. Лучшая точность на валидации: {best_val_acc:.1f}%")
    print(f"Модель сохранена: {SAVE_DIR / 'best_model.pth'}")


if __name__ == '__main__':
    train()
