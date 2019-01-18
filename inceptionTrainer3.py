from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils
import pretrainedmodels
import numpy as np
from PIL import Image
import glob
import cv2
import pandas as pd
import re
from torch import nn, Tensor, optim
import os
import csv
import datetime
import matplotlib.pyplot as plt
import torch
import matplotlib
matplotlib.use('agg')
import io
# import matplotlib.pyplot as plt


def plot_training_val(tr_loss_filename, val_loss_filename):
    now = datetime.datetime.now()
    now_date = (str(now).split(' ')[0]).replace('-', '_')
    with open(tr_loss_filename) as csvfile:
        reader = csv.reader(csvfile)
        trloss = list(reader)[0]
    with open(val_loss_filename) as csvfile:
        reader = csv.reader(csvfile)
        valloss = list(reader)[0]
    trloss = np.around(np.array(trloss, dtype='Float32'))
    valloss = np.around(np.array(valloss, dtype='Float32'))

    plt.plot(range(len(trloss)), trloss, label='Training Loss')
    plt.plot(range(len(valloss)), valloss, label='Validation Loss')
    plt.title('Training Loss and Validation Loss')
    plt.xlabel('Iterations over 10 Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig(now_date + 'TrainingValLoss.png')
    return trloss, valloss


def calculate_val_loss(model, val_data, targets, batch_size):
    v_idx = np.arange(0, len(val_data), batch_size)
    vloss = 0
    guesses = []
    true_age = []
    val_loss = []
    for k in range(len(v_idx)):
        val_outputs = model(Tensor(np.concatenate(
            val_data[v_idx[k]:batch_size-1, :, :, :])).to(device))
        vloss += criterion(val_outputs,
                           Tensor([targets[v_idx[k]:batch_size-1]]).view(-1, 1).to(device))
        guesses.extend(val_outputs)
        true_age.extend(targets[v_idx[k]:batch_size-1])
    val_loss.append(vloss.item()/len(v_idx))
    return val_loss, guesses, true_age


def clahe_augment(img):
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

class BoneageDataset(Dataset):
    """Bone Age dataset."""

    def __init__(self, csv_file, root_dir, transform = False):
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
        img_name = os.path.join(self.root_dir,str(self.landmarks_frame.iloc[idx, 0])+'.png')
        image = cv2.imread(img_name,0)
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
        return sample

def load_data(training_folder, testing_folder, training_labels, testing_labels):
    training_list = []
    tr_labels = []
    for filename in glob.glob(training_folder+'/*.png'):
        im = cv2.imread(filename)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # resize
        im = cv2.resize(im, (299, 299))

        # scale
        means = [np.mean(im[:, :, i]) for i in range(im.shape[2])]
        stds = [np.std(im[:, :, i]) for i in range(im.shape[2])]
        im = np.divide((im - means), stds)

        im = np.expand_dims(im.reshape([3, 299, 299]), axis=0)
        training_list.append(im)
        tr_labels.append(training_labels['boneage']
                         [int(re.findall(r'\d+', filename)[0])])
    rperm = np.random.permutation(len(tr_labels))
    tr_labels_all = np.array(tr_labels)[rperm]
    training_list_all = np.array(training_list)[rperm]

    val_perc = 0.9
    val_labels = tr_labels_all[round(len(tr_labels_all)*val_perc):]
    val_data = training_list_all[round(len(tr_labels_all)*val_perc):]

    tr_labels = tr_labels[:round(len(tr_labels_all)*val_perc)]
    training_list = training_list_all[:round(len(tr_labels_all)*val_perc)]
    return training_list, val_data, tr_labels, val_labels


# def train_model(model, training_list, val_data, trlabels, val_labels, EPOCHS, criterion, optimizer):
def train_model(model, trainloader, valloader, EPOCHS, criterion, optimizer, gender = False):
    now = datetime.datetime.now()
    now_date = (str(now).split(' ')[0]).replace('-', '_')

    epoch_loss = []

    # batch_size = 10
    # batch_list = [concatenate(training_list[i:i+(batch_size-1)) for i in np.arange(0,len(training_list)-batch_size,batch_size)]
    iter_loss = []
    iter = 0
    val_loss = []
    for epoch in range(EPOCHS):
        running_loss = 0
        for i, (inputs, labels) in enumerate(trainloader):
            # get the inputs
            # inputs, labels = data
        # for i, tr_im in enumerate(training_list):
            labels = labels[:,:,0]
            optimizer.zero_grad()

            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.float().to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if np.mod(iter, 1000) == 0:
                iter_loss.append(running_loss/(i+1))
                print(['EPOCH ' + str(epoch) + ', iter ' +
                       str(i+1) + ', Loss: ' + str(running_loss/(i+1))])
                vloss = 0
                for j,(val_input, val_labels) in enumerate(valloader):
                    # v_idx = np.random.randint(0, len(val_data), 10)
                    val_outputs = model(val_input.to(device))
                    val_labels = val_labels[:,:,0]
                    vloss += criterion(val_outputs, val_labels.float().to(device)).item()
                val_loss.append(vloss/(j+1))
                print(['Validation Loss: ' + str(val_loss[-1])])
                if len(val_loss) > 2 and val_loss[-1] > val_loss[-2]:
                    torch.save(model.state_dict(), os.getcwd() +
                               '/' + now_date + 'inception_model.pt')

            iter += 1

        print(running_loss/(i+1))
        epoch_loss.append(running_loss/(i+1))
    return iter_loss, val_loss


# training_folder = 'Data/boneage-training-dataset'
# testing_folder = 'Data/boneage-test-dataset'
# labels_csv = 'Data/boneage-training-dataset.csv'
# labels_csv_tst = 'Data/boneage-test-dataset.csv'

training_folder = 'SampleData/'
testing_folder = 'SampleDataTest/'
labels_csv = 'SampleData.csv'
labels_csv_tst = 'SampleDataTest.csv'

training_labels = pd.read_csv(
    labels_csv, index_col=0).to_dict()
testing_labels = pd.read_csv(
    labels_csv_tst, index_col=0).to_dict()

data_transform = transforms.Compose([
    transforms.Resize((299,299)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(10),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0, 0, 0],
                         std=[1, 1, 1])
])

transformed_dataset = BoneageDataset(csv_file=labels_csv,
                                     root_dir=training_folder,
                                     transform=data_transform)
validation_split = .8
indices = list(range(len(transformed_dataset)))
split = int(np.floor(validation_split * len(transformed_dataset)))
np.random.seed(0)
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

trainloader = DataLoader(transformed_dataset, batch_size=4,
                        num_workers=0, sampler= train_sampler)
valloader = DataLoader(transformed_dataset, batch_size = 4,
                        num_workers=0, sampler = train_sampler)

# training_list, val_data, trlabels, val_labels = load_data(
#     training_folder, testing_folder, training_labels, testing_labels)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Load model
model_name = 'inceptionv4'
model = pretrainedmodels.__dict__[model_name](
    num_classes=1000, pretrained='imagenet').to(device)

# Change last layer to linear layer
model.last_linear = nn.Linear(
    in_features=1536, out_features=1, bias=True).to(device)
# Need to get rid of logits layer too
# def forward(model, input):
#     x = self.features(input)
#     return x

# Possibly need to change input size too
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
criterion = nn.MSELoss()
EPOCHS = 10
iterloss, valloss = train_model(model, trainloader, valloader, EPOCHS, criterion, optimizer)

now = datetime.datetime.now()
now_date = (str(now).split(' ')[0]).replace('-', '_')
with open(now_date + 'iter_loss.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(iterloss)

with open(now_date + 'val_loss.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(valloss)

model_name = 'inceptionv4'
model_loaded = pretrainedmodels.__dict__[model_name](
    num_classes=1000, pretrained='imagenet').to(device)
# Change last layer to linear layer
model_loaded.last_linear = nn.Linear(
    in_features=1536, out_features=1, bias=True)
model_loaded.load_state_dict(torch.load(
    'inception_model_110.pt', map_location='cpu'))

with torch.no_grad():
    loss, guesses, true_age = calculate_val_loss(
        model_loaded, val_data, val_labels, 10)

plt.scatter(guesses, true_age)
plt.show()
