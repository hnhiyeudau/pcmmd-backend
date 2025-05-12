import torch
from torchvision import transforms
from PIL import Image
import io

model = torch.load("model/yolo10l_final.pt", map_location=torch.device('cpu'), weights_only=False)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

async def predict_image(file):
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()

    label = "Plasma Cell" if predicted.item() == 1 else "Non-Plasma Cell"
    return {"label": label, "confidence": round(confidence * 100, 2)}
