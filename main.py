import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models import FramePredictor
from utils.processing import MovingMNIST
from utils.train import Trainer
from PIL import Image

def save(out, i: int):
    out = out[0, [0]].cpu().detach().numpy() * 255
    out = out.repeat(3, axis=0).astype(np.uint8)
    out = out.transpose([1, 2, 0])
    image = Image.fromarray(out)
    image.save("./images/image_{}.png".format(i))

DEVICE = "cpu"

dataset = MovingMNIST("../data")
X, Y = dataset[0]
print(X.shape, Y.shape)
print(X.dtype, Y.dtype)

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset, batch_size=256, shuffle=False)

model = FramePredictor(input_size=(1, 64, 64), kernel_size=3, hidden_size=32).to(DEVICE)
optimizer = optim.Adam(params=model.parameters(), lr=0.0005)
loss_fn = nn.BCELoss()
trainer = Trainer(
    model=model, train_loader=train_loader, val_loader=val_loader,
    optimizer=optimizer, loss_fn=loss_fn, epochs=2, filepath="./saved_models/model.pt")

for X, Y in train_loader:
    model(X)
    break
# trainer.run()