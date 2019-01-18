import torch
import pretrainedmodels
import numpy as np
from PIL import Image
import glob
import cv2
import pandas as pd
# import matplotlib.pyplot as plt
import re
from torch import nn, Tensor, optim
import os
import csv
import datetime
import matplotlib.pyplot as plt
import torch


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


def train_model(model, training_list, val_data, trlabels, val_labels, EPOCHS, criterion, optimizer):

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
        for i, tr_im in enumerate(training_list):
            optimizer.zero_grad()

            outputs = model(Tensor(tr_im).to(device))
            loss = criterion(outputs, Tensor([[tr_labels[i]]]).to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if np.mod(iter, 1000) == 0:
                iter_loss.append(running_loss/(i+1))
                print(['EPOCH ' + str(epoch) + ', iter ' +
                    str(i+1) + ', Loss: ' + str(running_loss/(i+1))])
                v_idx = np.random.randint(0, len(val_data), 10)
                val_outputs = model(
                    Tensor(np.concatenate(val_data[v_idx])).to(device))
                vloss = criterion(val_outputs, Tensor(
                    [val_labels[v_idx]]).view(-1, 1).to(device))
                val_loss.append(vloss.item())
                print(['Validation Loss: ' + str(val_loss[-1])])
                if len(val_loss) > 2 and val_loss[-1] > val_loss[-2]:
                    torch.save(model.state_dict(), os.getcwd() +
                            '/' + now_date + 'inception_model.pt')

            iter += 1

        print(running_loss/(i+1))
        epoch_loss.append(running_loss/(i+1))
    return iter_loss, val_loss


training_folder = 'SampleData'
testing_folder = 'SampleDataTest'

training_labels = pd.read_csv(
    'boneage-training-dataset.csv', index_col=0).to_dict()
testing_labels = pd.read_csv(
    'boneage-test-dataset.csv', index_col=0).to_dict()


training_list, val_data, trlabels, val_labels = load_data(
    training_folder, testing_folder, training_labels, testing_labels)

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

# iter_loss, val_loss = train_model(model, training_list, val_data, trlabels,
#             val_labels, EPOCHS, criterion, optimizer)
now = datetime.datetime.now()
now_date = (str(now).split(' ')[0]).replace('-', '_')
with open(now_date + 'iter_loss.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(iter_loss)

with open(now_date + 'val_loss.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(val_loss)

trloss, valloss = plot_training_val('iter_loss.csv', 'val_loss.csv')
plt.show()

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
        model_loaded, val_data, val_labels, 2)

plt.scatter(guesses, true_age); plt.show()