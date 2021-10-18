from ctypes import resize
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.models
from torchsummary import summary
import numpy as np
from PIL import Image
from tqdm import tqdm

# from Models import MyClassifier
import util


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
NUM_CLASSES = 201  # 200+1 (ignore index 0)
EPOCH = 3
BATCH_SIZE = 8
LR = 0.001


class MyDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        dataset = []
        with open(txt_file, 'r') as file_handler:
            for line in file_handler:
                line = line.strip('\n')
                imgName, label = line.split()
                label = label.split('.')[0]
                dataset.append((imgName, label))

        self.dataset = dataset
        self.transform = transform
        
    def __getitem__(self, index):
        imgName, label = self.dataset[index]
        img = Image.open('data/training_images/'+imgName).convert('RGB')

        if self.transform:
            img = self.transform(img)
        
        # Convert 'str' label to one-hot vector?
        label = int(label)
        label = torch.tensor([label])

        return img, label
    
    def __len__(self):
        return len(self.dataset)


train_transform = transforms.Compose([
        # transforms.RandomRotation(degrees=3),
        # transforms.RandomHorizontalFlip(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

train_dataset = MyDataset(txt_file='data/training_labels.txt', transform=train_transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)




# Initialize a model, and put it on the device specified.
# model = MyClassifier().to(device)
# model.device = device

# Init well-known model and modify the last FC layer
model = torchvision.models.resnet18(pretrained=True)
# # Freeze all the network except the final layer
# # Ps: Only need to do this when feature extracting (We are finetuning now, no need to do this!) 
# # for param in model.parameters():
# #     param.requires_grad = False # Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model = model.to(device)  # Send the model to GPU

summary(model, (3, 32, 32))
num_of_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('num_of_params:', num_of_params)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



# Train the model
curr_lr = LR
best_acc = 0
best_epoch = 0
training_loss_history, training_accuracy_history = [], []
val_loss_history, val_accuracy_history = [], []
val_top3error_history = []

for epoch in tqdm(range(EPOCH)):
    # ---------- Training ---------
    model.train()  # Set model to training mode
    running_loss = 0.0
    running_corrects = 0
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.set_grad_enabled(True):
            # Forward pass
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            # loss = criterion(outputs, labels)  # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            # loss = criterion(outputs, torch.max(labels, 1)[1])
            loss = criterion(outputs, labels.squeeze())
            
            # zero the parameter gradients
            optimizer.zero_grad()
            # Backward and optimize (only in train phase)
            loss.backward()
            optimizer.step()
        
        # statistics
        running_loss += loss.item() * images.size(0)  # images.size(0) means BATCH_SIZE
        running_corrects += torch.sum(preds == labels.squeeze().data)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)
    training_loss_history.append(epoch_loss)
    training_accuracy_history.append(epoch_acc)

    # print training info
    print('Training Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    # Decay learning rate
    if (epoch+1) % 10 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)


    # # ---------- Validation ----------
    # model.eval()  # Set model to evaluate mode
    # running_loss = 0.0
    # running_corrects = 0
    # running_top3_errors = 0
    # softmax = nn.Softmax(dim=-1)  # Define Softmax function

    # for i, (images, labels) in enumerate(tqdm(val_loader)):
    #     images = images.to(device)
    #     labels = labels.to(device)
        
    #     with torch.set_grad_enabled(False):         
    #         outputs = model(images)
    #         _, preds = torch.max(outputs, 1)
            
    #         loss = criterion(outputs, labels)
            
    #     probabilities = softmax(outputs)
    #     _, indices = torch.sort(probabilities, descending=True)
    #     top3_preds = indices[:, :3]  # Top-3 predicted class

    #     # count top3 errors
    #     for row, label in zip(top3_preds, labels):
    #         if label not in row:
    #             running_top3_errors += 1

    #     # statistics
    #     running_loss += loss.item() * images.size(0)
    #     running_corrects += torch.sum(preds == labels)


    # epoch_loss = running_loss / len(val_dataset)
    # epoch_acc = running_corrects.double() / len(val_dataset)
    # epoch_top3error = running_top3_errors / len(val_dataset)
    # val_loss_history.append(epoch_loss)
    # val_accuracy_history.append(epoch_acc)
    # val_top3error_history.append(epoch_top3error)

    # if epoch_acc > best_acc:
    #     best_acc = epoch_acc
    #     best_epoch = epoch
    #     torch.save(model, 'output/best_model.pth')
    
    # # print validation info
    # print('Val Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))


# Output the training info cruves
# util.save_loss_history(training_loss_history, val_loss_history, EPOCH)
# util.save_accuracy_history(training_accuracy_history, val_accuracy_history, EPOCH)
# util.save_top3error_history(val_top3error_history, EPOCH)
# print('Best accuracy in validation:', best_acc)
# print('Best epoch:', best_epoch)
# print('Top3 error rate of validation data:', val_top3error_history)

torch.save(model, 'output/last_model.pth')