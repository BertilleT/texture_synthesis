import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
#transforms
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from utils import *
from model import *

# Define parameters for target image
model_path = '../model/vgg_conv.pth'
img = '../img/tissu.png'    
img_size = 512
target = download_img(img, img_size).to(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
layers = [1, 6, 11, 20, 29]
layers_weights = [1/n**2 for n in [64,128,256,512,512]]

# Def parameters for model
loss = nn.MSELoss()
epochs = 5#1000
random_img = torch.randn_like(target)
optimizer = optim.LBFGS([random_img])
random_img.requires_grad=True
#print random_img and target size
#print(random_img.size())
#print(target.size())

# ---------------------------------------------------- Def Gram matrices of target image
# Def model cnn
cnn = models.vgg19().features.to(device)
pretrained_dict = torch.load(model_path)
for param, item in zip(cnn.parameters(), pretrained_dict.keys()):
    param.data = pretrained_dict[item].type(torch.FloatTensor).to(device)
cnn.requires_grad_(False)

features_maps = {}
# Def hook to save features maps of some chosen layers at the forward pass
for layer in layers:
    handle = cnn[layer].register_forward_hook(save_features_map(layer, fm=features_maps))

# Def target Gram matrix for some specific layers
cnn(target)
Gram_target_features_maps = [gram(features_maps[key]) for key in layers] 

# ---------------------------------------------------- Def Gram matrices of synth image
synth_img = textureGeneration(epochs, random_img, cnn, loss, target, Gram_target_features_maps, features_maps, layers, optimizer)

'''#plot target and synth images
target = target.cpu().detach().squeeze(0).permute(1, 2, 0).numpy()
synth_img = synth_img.cpu().detach().squeeze(0).permute(1, 2, 0).numpy()

#synth_img = synth_img.cpu().squeeze(0).permute(1, 2, 0)
#plot target and synth images
plt.figure()
plt.imshow(target)
plt.figure()
plt.imshow(synth_img)
plt.show()'''