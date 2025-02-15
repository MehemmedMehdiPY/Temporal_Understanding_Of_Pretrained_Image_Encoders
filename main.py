import numpy as np
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
from models import FramePredictor
from PIL import Image

def save(out, i: int):
    out = out[0, [0]].cpu().detach().numpy() * 255
    out = out.repeat(3, axis=0).astype(np.uint8)
    out = out.transpose([1, 2, 0])
    image = Image.fromarray(out)
    image.save("./images/image_{}.png".format(i))

DEVICE = "cuda"

image = np.load("../data/sample.npy")
print(np.unique(image))

lim = 150 / 255
image = (image - 0) / (255 - 0)
image[image >= lim] = 1
image[image < lim] = 0

image = torch.tensor(image[None, :, None, :]).to(torch.float32)
x = image[:, :3].to(DEVICE)
y = image[:, 3].to(DEVICE)

print(x.shape, y.shape)

model = FramePredictor(input_size=(1, 64, 64), kernel_size=3, hidden_size=64).to(DEVICE)
optimizer = optim.Adam(params=model.parameters(), lr=0.0005)
loss_fn = nn.BCELoss()

for i in range(1000):
    optimizer.zero_grad()
    o = model(x)
    loss = loss_fn(o, y)
    loss.backward()
    optimizer.step()
    loss_item = loss.item()
    save(o, i)

    if i % 100 == 0:
        print(i)

checkpoints = model.state_dict()
torch.save(checkpoints, "./saved_models/trial_model.pt")
