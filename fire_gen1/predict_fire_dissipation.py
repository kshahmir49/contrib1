import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from torchsummary import summary
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from math import log10, sqrt
import argparse
from FireNet import FireNet
device = "mps" if torch.backends.mps.is_available() else "cpu"


def predict_fire_dissipation(input, time, aoi_size):
    model = FireNet()
    model.load_state_dict(torch.load("dm_unet_10K"))
    input = torch.from_numpy(input)
    input = input.type(torch.float32)
    input = input.resize_(1, 1, aoi_size, aoi_size)
    model = model.to(device)
    input = input.to(device)
    time = time.to(device)
    train_tensor = TensorDataset(input, time)
    y_pred = model(train_tensor[0], train_tensor[1])
    return y_pred


# def get_loss(model, x, y, t):
#     y_pred = model(x, t)
#     return F.l1_loss(y, y_pred)   

# def main():  
#     parser=argparse.ArgumentParser(description='fire prediction')
#     parser.add_argument('--sample_size',type=int,default=10000)
#     parser.add_argument('--load_pretrained', type=int, default=1)

#     args = parser.parse_args()
#     load_pretrained = bool(args.load_pretrained)

#     input = np.load("input.npy")
#     output = np.load("output.npy")
#     sensor_values = np.load("sensor_values.npy")

#     input = torch.from_numpy(input)
#     output = torch.from_numpy(output)
#     sensor_values = torch.from_numpy(sensor_values)
#     # last column
#     time = sensor_values[:,-1]
#     input, output, time = input.type(torch.float32), output.type(torch.float32), time.type(torch.float32)


#     input = input.resize_(input.shape[0], 1, input.shape[1], input.shape[2])
#     output = output.resize_(output.shape[0], 1, output.shape[1], output.shape[2])
#     # time = time.resize_(time.shape[0], 1)


#     model = FireNet()

#     model = model.to(device)
#     input = input.to(device)
#     output = output.to(device)
#     time = time.to(device)
#     train_tensor = TensorDataset(input, time, output)
#     train_dataloader = DataLoader(train_tensor, batch_size=50)
#     optimizer = Adam(model.parameters(), lr=0.001)
#     epochs = 2000 # Try more!

#     if not load_pretrained:

#         for epoch in range(epochs):
#             for step, batch in enumerate(train_dataloader):
#                 optimizer.zero_grad()
#                 t = batch[1]
#                 loss = get_loss(model, batch[0], batch[2], t)
#                 loss.backward()
#                 optimizer.step()

#                 if epoch % 5 == 0 and step == 0:
#                     print(f"Epoch {epoch} | Loss: {loss.item()} ")

#         torch.save(model.state_dict(), "dm_unet_10K")
    
#     else:
#         model.load_state_dict(torch.load("dm_unet_10K"))
#         psnr_sum = 0
#         with torch.no_grad():
#             for step, batch in enumerate(train_dataloader):
#                 y_pred = model(batch[0], batch[1])
#                 for i in range(y_pred.shape[0]):
#                     squared_sum = ((batch[2][i].resize_(batch[2][i].shape[1], batch[2][i].shape[2])
#                                     - y_pred[i].resize_(y_pred[i].shape[1], y_pred[i].shape[2])) ** 2).cpu().numpy()
#                     mse = np.mean(squared_sum)
#                     if(mse == 0):  # MSE is zero means no noise is present in the signal .
#                             # Therefore PSNR have no importance.
#                         print("The prediction is perfect")
#                     max_pixel = 1.0
#                     psnr = 20 * log10(max_pixel / sqrt(mse))
#                     psnr_sum += psnr
#                     print("PSNR is: ", psnr)
#         print("The average psnr is ", psnr_sum/output.shape[0])


# if __name__=='__main__':
#     main()