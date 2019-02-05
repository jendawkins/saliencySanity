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
RANDOMIZED_PARAMS = False
RANDOMIZED_LABELS = False
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
        image = Image.fromarray(image)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        if self.transform:
            image = self.transform(image)
        sample = (image, landmarks)
        return sample


training_folder = 'SampleData'
labels_csv = 'SampleData.csv'

data_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0, 0, 0],
                         std=[1, 1, 1])
])

transformed_dataset = BoneageDataset(csv_file=labels_csv,
                                     root_dir=training_folder,
                                     transform=data_transform)
trainloader = DataLoader(transformed_dataset, batch_size=1,
                         num_workers=0, shuffle = True)

prep_img, target_class = next(iter(trainloader))
target_class = target_class[:, :, 0]

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model_name = 'inceptionv4'
model = pretrainedmodels.__dict__[model_name](
    num_classes=1000, pretrained='imagenet').to(device)
# Change last layer to linear layer
model.last_linear = nn.Linear(
    in_features=1536, out_features=1, bias=True)
model.load_state_dict(torch.load(MOD_NAME, map_location='cpu'))

orig_im = np.transpose(prep_img.squeeze(),(1,2,0)).numpy()
orig_im = Image.fromarray((orig_im*255).astype('uint8'))
file_name_to_export = str(transformed_dataset.data_num)
if RANDOMIZED_LABELS:
    file_name_to_export += '_RAND_LABS'
    s1 = target_class.data[0][0].numpy() - 50 
    s2 = target_class.data[0][0].numpy() + 50
    sample_data = np.concatenate((np.arange(0,s1), np.arange(s2,200)))
    target_class = torch.Tensor([np.random.choice(sample_data)]).view(target_class.shape)

if RANDOMIZED_PARAMS:
    file_name_to_export += '_RAND_PARAMS'
    model = pretrainedmodels.__dict__[model_name](
        num_classes=1000, pretrained='imagenet').to(device)
    model.last_linear = nn.Linear(
        in_features=1536, out_features=1, bias=True)

def normalize_im(image):
    if image.shape[0] <= 3:
        image = np.transpose(image, (1, 2, 0))
    image = (image - image.min())
    image /= image.max()
    return image


def red_map(im):
    if im.shape[0] == 3:
        return 'Need grayscale image'
    cm = plt.get_cmap('Reds')
    if len(im.shape)>2:
        im = cm(im[:,:,0])
    else:
        im = cm(im)
    return im[:,:,:3]

def save_as_red_image(img, filename, percentile=99):
    img_2d = np.sum(img, axis=0)
    span = abs(np.percentile(img_2d, percentile))
    vmin = -span
    vmax = span
    img_2d = np.clip((img_2d - vmin) / (vmax - vmin), -1, 1)
    img_2d = red_map(img_2d)
    cv2.imwrite(filename, img_2d * 255)
    return

def save_grayscale_im(img, filename):
    img = red_map(normalize_im(img))*255
    img = Image.fromarray(img.astype(np.uint8))
    img.save('../results/'+filename)


"""
Vanilla Gradients
"""
# import pdb; pdb.set_trace()
## VANILLA BACKPROP
prep_img.requires_grad = True
target_class.requires_grad = True
vanilla_grad = VanillaGrad(model)
vanilla_saliency, output = vanilla_grad(prep_img, index=target_class.float())
fname_addn = '_Pred' + str(int(output.detach().item())) + '_True' + str(int(target_class.detach().item()))
# save_gradient_images(vanilla_saliency, file_name_to_export + '_Vanilla_BP_color')
grayscale_vanilla_grads = convert_to_grayscale(vanilla_saliency)
print(file_name_to_export)
save_grayscale_im(grayscale_vanilla_grads,
                  file_name_to_export + fname_addn + '_Vanilla_BP_reds.png')

# Method 2:
VBP = vanilla_backprop.VanillaBackprop(model)
# Generate gradients
vanilla_grads = VBP.generate_gradients(prep_img, target_class.float())
grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)
save_grayscale_im(grayscale_vanilla_grads,
                  file_name_to_export + fname_addn + '_Vanilla_BP_reds_method1.png')
print('Vanilla backprop completed')


"""
Grad Cam
"""
## GRAD CAM
grad_cam = gradcam.GradCam(model, target_layer=21)
# Generate cam mask
cam = grad_cam.generate_cam(prep_img, target_class.float())
# Save mask
save_class_activation_images(orig_im, cam, file_name_to_export+'gradcam_METHOD1')

grad_cam = GradCam(
    pretrained_model=model,
    target_layer_names='21')

# # Compute grad cam
mask, output = grad_cam(prep_img, target_class.float())
fname_addn = '_Pred' + str(int(output.detach().item())) + \
    '_True' + str(int(target_class.detach().item()))
save_class_activation_images(orig_im, mask, file_name_to_export+fname_addn + 'gradcam')
# save_class_activation_images(orig_im, mask, file_name_to_export+'gradcam_method2')
# import pdb; pdb.set_trace()
# import pdb; pdb.set_trace()

mask = cam
print('Grad cam completed')


""" Gradient x Input """
input_im = prep_img.squeeze().detach().numpy()
grad_input = np.multiply(vanilla_saliency, input_im)
# save_gradient_images(
#     grad_input, file_name_to_export + '_gradTimesIn_color')
grayscale_grad_input = convert_to_grayscale(grad_input)
save_grayscale_im(grayscale_grad_input,
                  file_name_to_export + '_gradTimesIn_reds.png')


input_im = prep_img.squeeze().detach().numpy()
grad_input = np.multiply(vanilla_grads, input_im)
# save_gradient_images(
#     grad_input, file_name_to_export + '_gradTimesIn_color')
grayscale_grad_input = convert_to_grayscale(grad_input)
save_grayscale_im(grayscale_grad_input,
                  file_name_to_export + '_gradTimesIn_reds_METHOD1.png')

print('Gradient x Input Completed')

""" Integrated gradients """


"""GUIDED BACKPROP"""
guided_grad = GuidedBackpropGrad(pretrained_model=model)
guided_saliency, output = guided_grad(prep_img, index=target_class.float())
fname_addn = '_Pred' + str(int(output.detach().item())) + \
    '_True' + str(int(target_class.detach().item()))
# save_gradient_images(
#     guided_saliency, file_name_to_export + '_guided_BP_color')
grayscale_guided_saliency = convert_to_grayscale(guided_saliency)
save_grayscale_im(grayscale_guided_saliency,
                  file_name_to_export + fname_addn+ '_guided_BP_reds.png')

# METHOD 2
# Guided backprop
GBP = guided_backprop.GuidedBackprop(model)
# Get gradients
guided_grads = GBP.generate_gradients(prep_img, target_class.long())
grayscale_guided_saliency = convert_to_grayscale(guided_grads)
save_grayscale_im(grayscale_guided_saliency,
                  file_name_to_export + fname_addn + '_guided_BP_reds_METHOD1.png')
print('Guided backprop completed')


"""Guided GradCAM"""
# Compute guided backpropagation
cam_mask = np.zeros(guided_saliency.shape)
for i in range(guided_saliency.shape[0]):
    cam_mask[i, :, :] = mask
cam_guided_backprop = np.multiply(cam_mask, guided_saliency)
# save_gradient_images(
#     cam_guided_backprop, file_name_to_export + '_GuidedGradCam_color')
grayscale_cam_guided_backprop = convert_to_grayscale(cam_guided_backprop)
save_grayscale_im(grayscale_cam_guided_backprop,
                  file_name_to_export + '_GuidedGradCam_reds.png')

# METHOD 2
cam_gb = guided_gradcam.guided_grad_cam(cam, guided_grads)
grayscale_cam_guided_backprop = convert_to_grayscale(cam_gb)
save_grayscale_im(grayscale_cam_guided_backprop,
                  file_name_to_export + '_GuidedGradCam_reds_METHOD1.png')
print('Guided GradCam Completed')

"""SmoothGrad"""
smooth_grad = SmoothGrad(pretrained_model=model)
smooth_saliency, output = smooth_grad(prep_img, index=target_class.float())
fname_addn = '_Pred' + str(int(output.detach().item())) + \
    '_True' + str(int(target_class.detach().item()))
# save_gradient_images(
#     smooth_saliency, file_name_to_export + '_SmoothGrad_color')
grayscale_smooth_saliency = convert_to_grayscale(smooth_saliency)
save_grayscale_im(grayscale_smooth_saliency,
                  file_name_to_export + fname_addn +'_SmoothGrad_reds.png')


# METHOD 2
param_n = 25
param_sigma_multiplier = 6.667
smooth_grad = generate_smooth_grad(VBP,  # ^This parameter
                                   prep_img,
                                   target_class.long(),
                                   param_n,
                                   param_sigma_multiplier)

grayscale_smooth_saliency = convert_to_grayscale(smooth_grad)
save_grayscale_im(grayscale_smooth_saliency,
                  file_name_to_export + fname_addn + '_SmoothGrad_reds_METHOD1.png')
print('SmoothGrad Completed')

save_image(prep_img.squeeze().detach().numpy(), '../results/' +
           file_name_to_export + '_Original.png')
