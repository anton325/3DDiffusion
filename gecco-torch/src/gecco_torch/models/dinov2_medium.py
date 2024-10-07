import torch
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
import matplotlib.pyplot as plt
import time
import os
import random
from PIL import Image
from tqdm import tqdm

import pandas as pd

def set_seed(no):
    torch.manual_seed(no)
    random.seed(no)
    np.random.seed(no)
    os.environ['PYTHONHASHSEED'] = str()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(100)

batch_size = 8

transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # mu, sigma
    ])

img = Image.open('gt.png').resize((224, 224))
img = np.array(img)[:,:,:3]
img = torch.tensor(img).type(torch.float32)
trainloader = [img.unsqueeze(0).permute(0, 3, 1, 2)]


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)

#training
train_embeddings = []

dinov2_vits14.eval()
with torch.no_grad():
  for data in tqdm(trainloader):
    image_embeddings_batch = dinov2_vits14(data.to(device))

    train_embeddings.append(image_embeddings_batch.detach().cpu().numpy())