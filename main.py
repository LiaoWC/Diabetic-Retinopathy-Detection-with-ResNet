import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms
import matplotlib.pyplot as plt
from typing import List
from torch.utils.data import DataLoader
from torchvision import models
from copy import deepcopy
import copy
from torch import nn, optim
import time
from datetime import datetime
from PIL import Image
import numpy as np
import pytz
import seaborn as sns
from sklearn.metrics import confusion_matrix

#
os.chdir('/home/ubuntu/lab3')

#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImgDataset(Dataset):
    def __init__(self, img_dir: str, img_name_csv: str, img_label_csv: str):
        self.img_dir = img_dir
        self.img_names = pd.read_csv(img_name_csv, header=None)[0].tolist()
        self.img_labels = pd.read_csv(img_label_csv, header=None)[0].tolist()
        # Check len of names and labels
        assert len(self.img_names) == len(self.img_labels)

        self.transform = transforms.Compose([
            # transforms.Resize(224),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            # transforms.RandomCrop(196),
            # Mean 1, 2, 3: -1.1180, -1.4863, -1.4968
            # Std 1, 2, 3: 1.1021598814759284, 0.7934357085248648, 0.5722392300281713
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transforms.Normalize(mean=[-1.1180, -1.4863, -1.4968],
                                 std=[1.1021598814759284, 0.7934357085248648, 0.5722392300281713])
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # img_data = read_image(os.path.join(self.img_dir, self.img_names[idx] + '.jpeg'))
        img_data = Image.open(os.path.join(self.img_dir, self.img_names[idx] + '.jpeg'))
        img_data = self.transform(img_data)
        return img_data, self.img_labels[idx]

class ImgDatasetTest(Dataset):
    def __init__(self, img_dir: str, img_name_csv: str, img_label_csv: str):
        self.img_dir = img_dir
        self.img_names = pd.read_csv(img_name_csv, header=None)[0].tolist()
        self.img_labels = pd.read_csv(img_label_csv, header=None)[0].tolist()
        # Check len of names and labels
        assert len(self.img_names) == len(self.img_labels)

        self.transform = transforms.Compose([
            transforms.Resize(224),
            # transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            # transforms.RandomCrop(196),
            # Mean 1, 2, 3: -1.1180, -1.4863, -1.4968
            # Std 1, 2, 3: 1.1021598814759284, 0.7934357085248648, 0.5722392300281713
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transforms.Normalize(mean=[-1.1180, -1.4863, -1.4968],
                                 std=[1.1021598814759284, 0.7934357085248648, 0.5722392300281713])
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # img_data = read_image(os.path.join(self.img_dir, self.img_names[idx] + '.jpeg'))
        img_data = Image.open(os.path.join(self.img_dir, self.img_names[idx] + '.jpeg'))
        img_data = self.transform(img_data)
        return img_data, self.img_labels[idx]


def get_train_test_dataloader(batch_size_):
    train_dataset = ImgDataset(img_dir='/home/ubuntu/lab3/data/',
                               img_name_csv='/home/ubuntu/lab3/train_img.csv',
                               img_label_csv='/home/ubuntu/lab3/train_label.csv')
    test_dataset = ImgDatasetTest(img_dir='/home/ubuntu/lab3/data/',
                              img_name_csv='/home/ubuntu/lab3/test_img.csv',
                              img_label_csv='/home/ubuntu/lab3/test_label.csv')
    return DataLoader(dataset=train_dataset, batch_size=batch_size_, shuffle=False), DataLoader(dataset=test_dataset,
                                                                                                batch_size=batch_size_,
                                                                                                shuffle=False)


# Training
def train_model(dataloader_train_valid, model_save_dir, model_, criterion_, optimizer_, num_epochs,
                device_, save_epoch_record: bool = False, name_suffix: str = '', sched=None):
    model_ = model_.to(device_)
    start = time.time()
    train_results = []
    valid_results = []
    best_model_wts = copy.deepcopy(model_.state_dict())
    best_acc = 0.0
    all_pred = []
    all_labels = []

    for epoch in range(num_epochs):
        print(
            'Epoch {}/{} {}'.format(epoch + 1, num_epochs,
                                    '(lr={})'.format(sched.optimizer.param_groups[0]['lr']) if sched else ''))
        print('-' * 10)

        start_elapsed = datetime.now()

        all_pred = []
        all_labels = []

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model_.train()  # Set model_ to training mode
            else:
                model_.eval()  # Set model_ to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            datasetsize = 0

            # Iterate over data.
            iteration = 1
            for inputs, labels in dataloader_train_valid[phase]:
                if iteration % 400 == 0:
                    print(f'{iteration}/{len(dataloader_train_valid[phase])}\r')

                datasetsize += labels.shape[0]

                inputs = inputs.float().to(device_)
                labels = labels.long()

                # print(labels)
                labels = labels.to(device_)

                # zero the parameter gradients
                optimizer_.zero_grad()

                # forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    # print("########### output ##########")
                    outputs = model_(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion_(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer_.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if phase == 'valid':
                    all_pred.append(preds.cpu().numpy())
                    all_labels.append(labels.data.cpu().numpy())

                iteration += 1

            epoch_loss = running_loss / datasetsize
            epoch_acc = running_corrects.double() / datasetsize

            if phase == 'train':
                train_results.append([epoch_loss, epoch_acc])
                if sched:
                    sched.step(epoch_loss)
            if phase == 'valid':
                valid_results.append([epoch_loss, epoch_acc])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model_ (Early Stopping) and Saving our model_, when we get best accuracy
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model_.state_dict())
                model_save_name = type(model_).__name__ + "_BestModel_" + name_suffix + ".pt"
                path = os.path.join(model_save_dir, model_save_name)
                torch.save(model_.state_dict(), path)
                torch.save(model_, os.path.join(model_save_dir, type(
                    model_).__name__ + "_" + name_suffix + f'_Epoch{epoch + 1}_' + datetime.now().strftime(
                    "%m%d_%H%M%S") + ".pt"))
                #
                # print(type(all_pred))
                # print(all_pred)
                # print(type(all_labels))
                # print(all_labels)
                # plt.figure(figsize=(6, 4))
                # cm = confusion_matrix(np.concatenate(all_labels), np.concatenate(all_pred), labels=np.arange(1, 11))
                # sn.heatmap(cm, annot=True, cmap=plt.cm.Blues,
                #            xticklabels=np.arange(1, 11),
                #            yticklabels=np.arange(1, 11), )
                # plt.show()

        end_elapsed = datetime.now()

        elapsed_time = end_elapsed - start_elapsed
        print("elapsed time: {}".format(elapsed_time))

        print()

    # Calculating time it took for model_ to train
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model_ weights
    model_.load_state_dict(best_model_wts)

    # Save records of each epoch
    if save_epoch_record:
        with open(os.path.join('epoch_record/', type(model_).__name__) + '_' + name_suffix + '.txt', 'w') as file:
            # First line: train loss
            # Second line: train acc
            # Third line: valid loss
            # Fourth line: valid acc
            train_loss = [item[0] for item in train_results]
            train_acc = [item[1].tolist() for item in train_results]
            valid_loss = [item[0] for item in valid_results]
            valid_acc = [item[1].tolist() for item in valid_results]
            file.write(' '.join(map(str, train_loss)) + '\n')
            file.write(' '.join(map(str, train_acc)) + '\n')
            file.write(' '.join(map(str, valid_loss)) + '\n')
            file.write(' '.join(map(str, valid_acc)) + '\n')

    return model_, train_results, valid_results, all_pred, all_labels


#############################################################################
# from tqdm import tqdm
# for i in tqdm(get_train_test_dataloader(batch_size_=1)[1]):
#     x = torch.cat(i[0][0])
# mean = torch.mean(x, dim=(0, 1))
# std = torch.std(x, dim=(0, 1))

#
LR = 0.0001
EPOCHS = 5
BATCH_SIZE = 4
MODEL_SAVE_DIR = 'saved_models/'

#
N_LAYERS = 18  # 18 or 50
PRETRAINED = True
suffix = ('Pretrained' if PRETRAINED else 'NonPretrained') + (
    'ResNet18' if N_LAYERS == 18 else 'ResNet50') + 'Lr' + str(LR).split('.')[-1] + 'BatchSize' + str(
    BATCH_SIZE) + '_main2'
tr_dataloader, te_dataloader = get_train_test_dataloader(batch_size_=BATCH_SIZE)

# Get model
# resnet18_pretrained = models.resnet18(pretrained=True)
# resnet18_non_pretrained = models.resnet18(pretrained=False)
model = models.resnet18(pretrained=PRETRAINED) if N_LAYERS == 18 else models.resnet50(pretrained=PRETRAINED)

# Lock params if pretrained # TODO: know if pretrained layers should be locked
# for param in model.parameters():
#     param.requires_grad = False
# Replace FC
model.fc = nn.Linear(in_features=512 if N_LAYERS == 18 else 2048, out_features=5, bias=True)

#
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
# optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.0001)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=0, factor=0.25,
                                                 min_lr=0.0000001, threshold=1e-4, cooldown=0, verbose=True)
# scheduler = None
model, tr_results, va_results, last_epoch_pred, last_epoch_labels = train_model(
    dataloader_train_valid={'train': tr_dataloader, 'valid': te_dataloader},
    model_save_dir=MODEL_SAVE_DIR,
    model_=model,
    criterion_=criterion,
    optimizer_=optimizer,
    num_epochs=EPOCHS,
    save_epoch_record=True,
    name_suffix=suffix,
    device_=device,
    sched=scheduler
)


#
backup_model = deepcopy(model)

#
model.eval()
test_dataloader = get_train_test_dataloader(batch_size_=BATCH_SIZE)[1]
n_correct = 0
n_data = 0
for i_, (sample_, label_) in enumerate(test_dataloader):
    sample_, label_ = sample_.to(device), label_.to(device)
    pred = torch.max(model(sample_), 1)[1]
    n_correct += torch.sum(torch.eq(label_, pred))
    if i_ % 100 == 0:
        print(f'{i_}/{len(test_dataloader)}')
    n_data += len(sample_)
test_acc = n_correct / n_data

torch.save(model, '7959.pt')


# # Plot acc-epoch
# colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
# # for model_name in ['EEGNet', 'DeepConvNet']:
# for model_name in ['ResNet50', 'ResNet18']:
#     max_epoch = 0
#     plt.figure(figsize=(8, 6))
#     plt.title(f'{model_name}')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy(%)')
#     # plt.yticks(np.arange(10)/10)
#     # plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))  # 2 decimal places
#     filenames = []
#     for root, dirs, files in os.walk('epoch_record'):
#         for file in files:
#             if model_name in file:
#                 filenames.append(os.path.join(root, file))
#     for i, filename in enumerate(filenames):
#         with open(filename, 'r') as file:
#             lines = file.read().split('\n')
#             train_acc = list(map(float, lines[1].split(' ')))
#             test_acc = list(map(float, lines[3].split(' ')))
#             max_epoch = max(len(train_acc), max_epoch)
#             plt.plot(np.arange(len(train_acc)) + 1, train_acc, label=filename.split('_')[-1].split('.')[0] + '_train',
#                      color=colors[i], linestyle='solid')
#             plt.plot(np.arange(len(test_acc)) + 1, test_acc, label=filename.split('_')[-1].split('.')[0] + '_test',
#                      color=colors[i], linestyle='dashed')
#     plt.xticks(np.arange(max_epoch) + 1)
#     plt.legend()
#     plt.savefig('output/' + model_name + '_AccEpoch.png', format='png')
#     plt.show()
#
# print('done')
#
#
# ###############
# def test_model(name__: str, n_layers_, file_abs_path, batch_size_=8):
def test_model(name__: str, file_abs_path, batch_size_=8, ret=False, state_dict=False, n_layers_=None):
    if state_dict:
        N_LAYERS_ = n_layers_
        loaded_model_ = models.resnet18() if N_LAYERS_ == 18 else models.resnet50()
        loaded_model_.fc = nn.Linear(in_features=512 if N_LAYERS_ == 18 else 2048, out_features=5, bias=True)
        loaded_model_.load_state_dict(torch.load(file_abs_path))
    else:
        loaded_model_ = torch.load(file_abs_path)
    loaded_model_.eval()
    loaded_model_ = loaded_model_.to(device)
    #
    _, test_dataloader_ = get_train_test_dataloader(batch_size_=batch_size_)
    n_correct_ = 0
    n_data_ = 0
    all_pred = []
    all_label = []
    for i__, (sample__, label__) in enumerate(test_dataloader_):
        sample__, label__ = sample__.to(device), label__.to(device).float()
        pred_ = torch.max(loaded_model_(sample__), 1)[1]
        n_correct_ += torch.sum(torch.eq(label__, pred_))
        if i__ % 100 == 0:
            print(f'{i__}/{len(test_dataloader_)}')
        n_data_ += len(sample__)
        all_pred.append(pred_)
        all_label.append(label__)
    test_acc_ = n_correct_ / n_data_
    print(f'{name__}: {test_acc_}')
    del loaded_model_
    torch.cuda.empty_cache()
    if ret:
        return all_label, all_pred


# labels, preds = test_model('18_Pretrained', '/home/ubuntu/lab3/resnet18_pretrained.pt', ret=True)
test_model('18_Pretrained', '/home/ubuntu/lab3/resnet18_pretrained.pt')
test_model('18_NonPretrained', '/home/ubuntu/lab3/resnet18_NonPretrained.pt')
test_model('50_Pretrained', '/home/ubuntu/lab3/resnet50_Pretrained.pt')
test_model('50_NonPretrained', '/home/ubuntu/lab3/ResNet_BestModel_NonPretrainedResNet50.pt', n_layers_=50,
           state_dict=True)

# # For confusion matrix
# y_pred = []
# for tensor in preds:
#     y_pred.extend(tensor.tolist())
# y_truth = []
# for tensor in labels:
#     y_truth.extend(tensor.tolist())
#
# plt.figure(figsize=(4, 4))
# cf = confusion_matrix(y_truth, y_pred) / float(len(y_truth))
# sns.heatmap(cf,
#             cmap="Blues", annot=True, fmt=".4f", cbar=False,
#             xticklabels=np.arange(5), yticklabels=np.arange(5))
# plt.title("Confusion Matrix")
# plt.show()
#
# #
# # test_model('18Pretrained', '/home/ubuntu/lab3/ResNet_PretrainedResNet18Lr0025BatchSize256_Epoch1_0802_152948.pt')
# # test_model('50Pretrained', 50, '/home/ubuntu/lab3/saved_models/ResNet_BestModel_PretrainedResNet50.pt')
# # test_model('50NonPretrained', 50, '/home/ubuntu/lab3/saved_models/ResNet_BestModel_NonPretrainedResNet50.pt')
# # test_model('18Pretrained', 18, '/home/ubuntu/lab3/saved_models/ResNet_BestModel_PretrainedResNet18.pt')
# # test_model('18NonPretrained', 18, '/home/ubuntu/lab3/saved_models/ResNet_BestModel_NonPretrainedResNet18.pt')
# # test_model('ResNet_BestModel_NonPretrainedResNet18Lr0025BatchSize8', 18,
# #            '/home/ubuntu/lab3/saved_models/ResNet_BestModel_NonPretrainedResNet18Lr0025BatchSize8.pt')
# # test_model('ResNet_BestModel_PretrainedResNet18MobaAddTransform', 18,
# #            '/home/ubuntu/lab3/saved_models/ResNet_BestModel_PretrainedResNet18MobaAddTransform.pt')
# # test_model('ResNet_BestModel_PretrainedResNet18MobaAddTransformToTensor', 18,
# #            '/home/ubuntu/lab3/saved_models/ResNet_BestModel_PretrainedResNet18MobaAddTransformToTensor.pt')
