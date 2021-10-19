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
from collections import defaultdict


# Mapping of labels: 0~199 to classNames: 1~200
label_mapping = dict()
i = 0
with open('data/classes.txt', 'r') as f:
    for line in f:
        line = line.strip('\n')
        label_mapping[i] = line
        i += 1


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
BATCH_SIZE = 8


class MyDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        data = []
        with open(txt_file, 'r') as file_handler:
            for line in file_handler:
                imgName = line.strip('\n')
                data.append(imgName)

        self.data = data
        self.transform = transform
        
    def __getitem__(self, index):
        imgName = self.data[index]
        img = Image.open('data/testing_images/'+imgName).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img
    
    def __len__(self):
        return len(self.data)

data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.483, 0.498, 0.432], std=[0.237, 0.233, 0.272])
    ])
test_dataset = MyDataset(txt_file='data/testing_img_order.txt', transform=data_transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False)

# Load model
model = torch.load('output/last_model.pth')
summary(model, (3, 32, 32))

# ---------- Testing: save Top1 predicted label to predictions list ----------
model.eval()
predictions = []
with torch.no_grad():
    total = 0
    for images in tqdm(test_loader):
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        for item in preds:
            predictions.append(item.tolist())

        total += images.size(0)
# print(predictions)


# Create submission file: answer.txt
submission = []
with open('data/testing_img_order.txt') as f:
     test_images = [x.strip() for x in f.readlines()]  # all the testing images

for imgName, pred in zip(test_images, predictions):
    predicted_class = label_mapping[pred]
    submission.append([imgName, predicted_class])

np.savetxt('output/answer.txt', submission, fmt='%s')
print(f'Finish saving predictions of {total} testing data to output/answer.txt !')
