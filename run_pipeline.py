import torch
from src.dataloader import get_dataloaders
from src.model import get_model
from src.train import train_model
from src.evaluate import evaluate_on_test

# Set paths
train_dir = "data/processed/train"
val_dir = "data/processed/val"
test_dir = "data/processed/test"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data
train_loader, val_loader, test_loader, class_names = get_dataloaders(train_dir, val_dir, test_dir)

# Build Model
model = get_model(num_classes=len(class_names))

# Train
train_model(model, train_loader, val_loader, device)

# Load best model and test
model.load_state_dict(torch.load("saved_models/best_model.pth"))
evaluate_on_test(model, test_loader, class_names, device)
        