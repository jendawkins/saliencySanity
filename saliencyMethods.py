# from pytorch_cnn_visualizations import src
from torchvision import models
import numpy as np
import sys
from torchvision import transforms, utils
sys.path.append(
    '/Users/jenniferdawkins/saliencySanity/pytorch_cnn_visualizations/src/')
import gradcam
import vanilla_backprop
import guided_backprop
import pretrainedmodels
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torch
import glob
import pandas as pd
import torch
import re
import cv2
import torch
from misc_functions import *
from PIL import Image
import matplotlib.pyplot as plt
sys.path.append(
    '/Users/jenniferdawkins/saliencySanity/pytorch-smoothgrad/')
import lib.gradients
from lib.gradients import *

MOD_NAME = '2019_01_162inception_model.pt'

""" 
Paper Explanations & Pytorch CNN Visualizations Corresponding Models
Gradient Explanation = Gradient Visualization with Vanilla backpropogation
    Egrad = DS/Dx
Gradient x Input = 


"""

"""
Load Data
"""

class BoneageDataset(Dataset):
    """Bone Age dataset."""

    def __init__(self, csv_file, root_dir, transform=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def clahe_augment(self, img):
        clahe_low = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        clahe_medium = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        clahe_high = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        img_low = clahe_low.apply(img)
        img_medium = clahe_medium.apply(img)
        img_high = clahe_high.apply(img)
        augmented_img = np.array([img_low, img_medium, img_high])
        augmented_img = np.swapaxes(augmented_img, 0, 1)
        augmented_img = np.swapaxes(augmented_img, 1, 2)

        return augmented_img

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.landmarks_frame.iloc[idx, 0])+'.png')
        self.data_num = self.landmarks_frame.iloc[idx, 0]
        image = cv2.imread(img_name, 0)
        image = self.clahe_augment(image)
        image = cv2.resize(image, (299, 299))
        # image = np.divide(image - np.mean(image), np.std(image))
        # import pdb; pdb.set_trace()
        image = Image.fromarray(image)
        # image = np.expand_dims(image.reshape([3, 299, 299]), axis=0)

        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        # sample = {'image': image, 'landmarks': landmarks}
        if self.transform:
            image = self.transform(image)
        sample = (image, landmarks)
        # orig_im = np.transpose(image, (1,2,0)).numpy()
        return sample


training_folder = 'SampleData'
labels_csv = 'SampleData.csv'

data_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomAffine(10),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0, 0, 0],
                         std=[1, 1, 1])
])

transformed_dataset = BoneageDataset(csv_file=labels_csv,
                                     root_dir=training_folder,
                                     transform=data_transform)
trainloader = DataLoader(transformed_dataset, batch_size=1,
                         num_workers=0)

prep_img, target_class = next(iter(trainloader))
target_class = target_class[:, :, 0]
# img_filename = str(transformed_dataset.data_num) + '.png'
# file_name_to_export = 'VisualizationsFolder/'

# training_labels = pd.read_csv(
#     'boneage-training-dataset.csv', index_col=0).to_dict()
# testing_labels = pd.read_csv(
#     'boneage-test-dataset.csv', index_col=0).to_dict()

# files = glob.glob(training_folder+'/*.png')
# # get random filename
# filename = files[int(np.random.randint(0, len(files)-1, 1))]
# im = cv2.imread(filename)
# im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# # resize
# orig_im = cv2.resize(im, (299, 299))
# im = orig_im
# # scale
# means = [np.mean(im[:, :, i]) for i in range(im.shape[2])]
# stds = [np.std(im[:, :, i]) for i in range(im.shape[2])]
# im = np.divide((im - means), stds)

# prep_img = np.expand_dims(im.reshape([3, 299, 299]), axis=0)
# target_class = training_labels['boneage'][int(re.findall(r'\d+', filename)[0])]

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model_name = 'inceptionv4'
model = pretrainedmodels.__dict__[model_name](
    num_classes=1000, pretrained='imagenet').to(device)
# Change last layer to linear layer
model.last_linear = nn.Linear(
    in_features=1536, out_features=1, bias=True)
model.load_state_dict(torch.load(MOD_NAME, map_location='cpu'))
# prep_img = torch.Tensor(prep_img)
# target_class = torch.Tensor([target_class])
orig_im = np.transpose(prep_img.squeeze(),(1,2,0)).numpy()
orig_im = Image.fromarray((orig_im*255).astype('uint8'))
# orig_im2 = Image.fromarray(orig_im.astype('uint8'))
file_name_to_export = str(transformed_dataset.data_num)

def normalize_im(image):
    if image.shape[0] <= 3:
        image = np.transpose(image, (1,2,0))
    image = (image - image.min())
    image /= image.max()
    return image


"""
Grad Cam
"""
## GRAD CAM
grad_cam = gradcam.GradCam(model, target_layer=21)
# Generate cam mask
cam = grad_cam.generate_cam(prep_img, target_class)
# Save mask
save_class_activation_images(orig_im, cam, file_name_to_export+'gradcam')
print('Grad cam completed')

"""
Vanilla Gradients
"""
# import pdb; pdb.set_trace()
## VANILLA BACKPROP
prep_img.requires_grad = True
target_class.requires_grad = True
vanilla_grad = VanillaGrad(model)
vanilla_saliency = vanilla_grad(prep_img, index=target_class.float())

# vanilla_image = np.transpose(vanilla_saliency, (1, 2, 0))
# vanilla_image = (vanilla_image - vanilla_image.min()) / \
#     (vanilla_image - vanilla_image.min()).max()
# im = Image.fromarray((vanilla_image* 255).astype(np.uint8))
# save images
# save_gradient_images(vanilla_saliency, file_name_to_export + '_Vanilla_BP_color')
grayscale_vanilla_grads = convert_to_grayscale(vanilla_saliency)
plt.imshow(grayscale_vanilla_grads[0,:,:], cmap = 'Reds')
plt.save('../results/'+file_name_to_export + '_Vanilla_BP_reds')
# save_gradient_images(grayscale_vanilla_grads,file_name_to_export + '_Vanilla_BP_gray')
print('Vanilla backprop completed')

import pdb; pdb.set_trace()
# # import pdb; pdb.set_trace()
# def custom_hook(module, grad_in, grad_out):
#     import pdb; pdb.set_trace()
#     return grad_in, grad_out

# first_layer1 = list(model.features._modules.items())[0][1]
# first_layer_conv = list(model.features._modules.items())[0][1]

# first_layer2 = list(model2.features._modules.items())[0][1]

# first_layer1.register_forward_hook(custom_hook)
# first_layer2.register_forward_hook(custom_hook)
# criterion = nn.MSELoss()
# model_output = model(prep_img)
# loss = criterion(model_output, target_class.float())
# loss.backward()

# # grad = list(model.parameters()).grad
# grad_ims = list(model.parameters())[0].grad
# gip= (grad_ims - grad_ims.min())/(grad_ims - grad_ims.min()).max()
# gip2 = [np.hstack(gip[i:(i+4),:,:,:]) for i in np.arange(0,32,4)]
# grad_ims_f = np.vstack(gip2)
# for param in model.parameters():
#     print(param.shape)
# VBP = vanilla_backprop.VanillaBackprop(model)
# # Generate gradients
# vanilla_grads = VBP.generate_gradients(prep_img, target_class)


## GUIDED BACKPROP
GBP = guided_backprop.GuidedBackprop(model)
# Get gradients
guided_grads = GBP.generate_gradients(prep_img, target_class)
# Save colored gradients
# save_gradient_images(
#     guided_grads, file_name_to_export + '_Guided_BP_color')
# Convert to grayscale
grayscale_guided_grads = convert_to_grayscale(guided_grads)
# Save grayscale gradients
save_gradient_orig_images(np.squeeze(
    grayscale_guided_grads), 
    cv2.resize(np.array(orig_im)[:, :, 1], grayscale_guided_grads.shape[1:3]),
    file_name_to_export + '_Guided_BP_gray')

# save_gradient_images(grayscale_guided_grads,
#                      file_name_to_export + '_Guided_BP_gray')
# Positive and negative saliency maps
pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
# Why are these 32x149x149???
pos_sal = np.pad(pos_sal,1,'constant',constant_values = 1)[1:-1,:,:]
pos_sal_cat = np.reshape(pos_sal, (4*pos_sal.shape[1], 8*pos_sal.shape[2]))
save_gradient_images(pos_sal_cat, file_name_to_export + '_pos_sal')
# save_gradient_images(neg_sal, file_name_to_export + '_neg_sal')
print('Guided backprop completed')
