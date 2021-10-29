import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.models
from torchsummary import summary
from PIL import Image
from tqdm import tqdm
from center_loss import CenterLoss
import math


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
NUM_CLASSES = 200
EPOCH = 40
BATCH_SIZE = 8
LR = 0.001


class MyDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        data = []
        with open(txt_file, 'r') as file_handler:
            for line in file_handler:
                line = line.strip('\n')
                imgName, label = line.split()
                label = label.split('.')[0]
                data.append((imgName, label))

        self.data = data
        self.transform = transform
        
    def __getitem__(self, index):
        imgName, label = self.data[index]
        # img = Image.open('data/training_images/'+imgName).convert('RGB')
        img = Image.open('data/training_images_augmented/'+imgName).convert('RGB')

        if self.transform:
            img = self.transform(img)
        
        # Convert 'str' label to tensor
        label = torch.tensor(int(label) - 1)  # label 0~199 matches class 1~200

        return img, label
    
    def __len__(self):
        return len(self.data)


train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(degrees=3),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.7),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # computed from ImageNet images
    ])

# train_dataset = MyDataset(txt_file='data/training_labels.txt', transform=train_transform)
train_dataset = MyDataset(txt_file='data/training_augmented_labels.txt', transform=train_transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)



# Init well-known model and modify the last FC layer
# model = torchvision.models.resnext101_32x8d(pretrained=True)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

model = torchvision.models.densenet161(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, NUM_CLASSES)

model = model.to(device)  # Send the model to GPU

# Show model summary
# summary(model, (3, 32, 32))  # resnet
summary(model, (3, 224, 224))  # densenet
num_of_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total number of params:', num_of_params)


# Loss and optimizer
criterion_CE = nn.CrossEntropyLoss()
criterion_CL = CenterLoss(num_classes=200, feat_dim=200, use_gpu=True)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.0001)
optimizer_centloss = torch.optim.SGD(criterion_CL.parameters(), lr=0.5)

# Consine lr
# t = 10  # warm-up epochs
# T = EPOCH  # First 10 epoch for warm-up, T eooch for cosine rate
# n_t = 0.5
# lambda1 = lambda epoch: (0.9*epoch / t+0.1) if epoch < t else  0.1  if n_t * (1+math.cos(math.pi*(epoch - t)/(T-t)))<0.1 else n_t * (1+math.cos(math.pi*(epoch - t)/(T-t)))
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# Train the model
curr_lr = LR
best_acc = 0
best_epoch = 0
training_loss_history, training_accuracy_history = [], []
val_loss_history, val_accuracy_history = [], []
val_top3error_history = []
alpha = 0.001  # alpha (float): weight for center loss


for epoch in tqdm(range(EPOCH)):
    # ---------- Training ---------
    model.train()  # Set model to training mode
    running_loss = 0.0
    running_corrects = 0
    
    print('\nCurrent lr:', get_lr(optimizer))
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.set_grad_enabled(True):
            # Forward pass
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            cross_entropy_loss = criterion_CE(outputs, labels)
            center_loss = criterion_CL(outputs, labels)
            loss =  center_loss * alpha + cross_entropy_loss
            # print(f'cross_entropy_loss: {cross_entropy_loss.item()}')
            # print(f'center_loss: {center_loss.item()}')
            # print(f'total weighted loss: {loss.item()}')
            
            # zero the parameter gradients
            optimizer.zero_grad()
            optimizer_centloss.zero_grad()
            
            # Backward and optimize (only in train phase)
            loss.backward()
            optimizer.step()
           
            # multiple (1./alpha) in order to remove the effect of alpha on updating centers
            for param in criterion_CL.parameters():
                param.grad.data *= (1./alpha)
            optimizer_centloss.step()
            
        
        # statistics
        running_loss += loss.item() * images.size(0)  # images.size(0) means BATCH_SIZE
        running_corrects += torch.sum(preds == labels)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)
    training_loss_history.append(epoch_loss)
    training_accuracy_history.append(epoch_acc)

    # print training info
    print('Training Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    # Decay learning rate
    if (epoch+1) > 5:
        if (epoch+1) % 5 == 0:
            curr_lr /= 2
            update_lr(optimizer, curr_lr)

    # scheduler.step()


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
# util.save_loss_history(training_loss_history, EPOCH)
# util.save_accuracy_history(training_accuracy_history, val_accuracy_history, EPOCH)
# util.save_top3error_history(val_top3error_history, EPOCH)
# print('Best accuracy in validation:', best_acc)
# print('Best epoch:', best_epoch)
# print('Top3 error rate of validation data:', val_top3error_history)

torch.save(model, 'output/models/densenet161_3.pth')
print('Finish training. The last model is saved in output/models folder.')