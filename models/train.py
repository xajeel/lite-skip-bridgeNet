import torch
import torch.nn as nn
import torch.optim as optim
from models.model import OptimizedModel
from utils.dataset import get_data_loaders
from utils.train_utils import train_model
from utils.test_utils import evaluate_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader, test_loader = get_data_loaders()
model = OptimizedModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

NUM_EPOCHS = 20
for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    scheduler.step(test_acc)
