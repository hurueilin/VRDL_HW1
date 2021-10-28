import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchsummary import summary
import numpy as np
from PIL import Image
from tqdm import tqdm
import util
from argparse import ArgumentParser
import os


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
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
test_dataset = MyDataset(txt_file='data/testing_img_order.txt', transform=data_transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False)

# Parser for loading model
# parser = ArgumentParser()
# parser.add_argument("-m", "--model", help="the model(.pth) you want to load in", dest="model", default="last_model.pth")
# args = parser.parse_args()

# Load model
# model = torch.load(f'output/models/{args.model}')
# summary(model, (3, 32, 32))

# Load all models in output/models folder
models = []
modelFiles = os.listdir('output/models')
print(f'There are {len(modelFiles)} models in model ensemble:')
for i, modelFile in enumerate(modelFiles):
    print(f'Model #{i}: {modelFile}')
    models.append(torch.load(f'output/models/{modelFile}'))



# ---------- Testing: save Top1 predicted label to predictions list ----------
print('======= Predictions Starts =======')
modelsPredictions = []
for i, model in enumerate(models):
    print(f'Model #{i} predicting...')
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
        
        modelsPredictions.append(predictions)
        print(f'Finish saving predictions of {total} testing data of model #{i}!')



print('Majority voting...')
submission = util.majorityVote(modelsPredictions)


# Create submission file: answer.txt
submission = []
with open('data/testing_img_order.txt') as f:
     test_images = [x.strip() for x in f.readlines()]  # all the testing images

for imgName, pred in zip(test_images, predictions):
    predicted_class = label_mapping[pred]
    submission.append([imgName, predicted_class])

np.savetxt('output/answer.txt', submission, fmt='%s')
print(f'Finish saving final predictions of {total} testing data to output/answer.txt !')
