# https://github.com/facebookresearch/dinov2/issues/2

import torch
from PIL import Image
import torchvision.transforms as T
# import hubconf

# dinov2_vits14 = hubconf.dinov2_vits14()

repo = 'facebookresearch/dinov2'  # Replace with the actual repository if different
dinov2_vits14 = torch.hub.load(repo, 'dinov2_vits14', source='github') 
img = Image.open('gt.png')

transform = T.Compose([
T.Resize(224),
T.CenterCrop(224),
T.ToTensor(),
T.Normalize(mean=[0.5], std=[0.5]),
])

img = transform(img)[:3].unsqueeze(0)

with torch.no_grad():
    features = dinov2_vits14(img, return_patches=True)

print(features.shape)
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.decomposition import PCA

# pca = PCA(n_components=3)
# pca.fit(features)

# pca_features = pca.transform(features)
# pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
# pca_features = pca_features * 255

# plt.imshow(pca_features.reshape(16, 16, 3).astype(np.uint8))
# plt.savefig('meta_dog_features.png')