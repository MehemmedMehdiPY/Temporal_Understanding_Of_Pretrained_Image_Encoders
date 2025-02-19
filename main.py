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

DEVICE = "cpu"

dataset = MovingMNIST("../data")

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset, batch_size=256, shuffle=False)

model = FramePredictor(input_size=(1, 64, 64), kernel_size=3, hidden_size=32).to(DEVICE)
optimizer = optim.Adam(params=model.parameters(), lr=0.0005)
loss_fn = nn.BCELoss()
trainer = Trainer(
    model=model, train_loader=train_loader, val_loader=val_loader,
    optimizer=optimizer, loss_fn=loss_fn, epochs=2, filepath="./saved_models/model.pt")
trainer.run()