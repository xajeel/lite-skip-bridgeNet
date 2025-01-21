import torch
from PIL import Image
from torchvision import transforms
from models.model import OptimizedModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict(image_path, model_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    model = OptimizedModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    return predicted.item()

# Example usage
# print(predict("test_image.jpg", "best_model.pth"))
