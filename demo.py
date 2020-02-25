import torch
from cnn import EfficientNet
from data import to_tensor
from PIL import Image

model = EfficientNet()
checkpoint = torch.load("b0.pth")
model.load_state_dict(checkpoint.pop("state_dict"))
model.eval()

img = Image.open("examples/download.jpeg").convert("L")
img = img.resize((32,32))
img = to_tensor(img)
img = img.unsqueeze(0)

with torch.no_grad():
    print(torch.nn.functional.softmax(model(img), 1).numpy())