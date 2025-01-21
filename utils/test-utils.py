import torch

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    epoch_test_loss = 0
    epoch_test_correct = 0
    epoch_test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            epoch_test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            epoch_test_correct += (predicted == labels).sum().item()
            epoch_test_total += labels.size(0)

    return epoch_test_loss / len(test_loader), epoch_test_correct / epoch_test_total
