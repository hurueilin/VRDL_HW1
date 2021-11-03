# VRDL HW1 - Bird Images Classification
## Introduction
![](https://i.imgur.com/a72LK9H.png)

In this assignment, we are given 6,033 bird images belonging to 200 bird species. There are 3,000 training images and 3,033 test images. This assignment is a fine-grained classification task, which means it is difficult to differentiate different bird species. Simply use ImageNet pretrained model with limited training data to do transfer learning cannot get a satisfied accuracy in testing. To tackle the fine-grained classification, I add Center Loss along with Softmax (Cross-Entropy) Loss as my loss function. I also experiment various backbone models such as ResNet, ResNeXt, DenseNet and EfficientNet. Finally, the best accuracy I can get is around 0.8 using EfficientNet-B4 as backbone.


## Environment
Hardware
* CPU: AMD Ryzen 5 3600 6-Core
* GPU: NVIDIA GeForce RTX 3070 8GB

Packages
* torch: 1.7.1+cu110
* torchvision: 0.8.2+cu110
* torch-summary: 1.4.4
* numpy: 1.19.5
* Center Loss: The `center_loss.py` in my project is copied from [this](https://github.com/KaiyangZhou/pytorch-center-loss) repository 
* [EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch): Install with `pip install efficientnet_pytorch`, and load a pretrained EfficientNet with:
    ```python
    from efficientnet_pytorch import EfficientNet
    model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=200)
    ```

## Training

To train the model, run this command:

```
python Train.py
```

## Reproducing Submission
1. Clone this project or download ZIP.


2. Create a folder named `data` and put `testing_images`, `classes.txt`, `testing_img_order.txt` inside.


3. Download the submitted [efficientnetb4_1.pth](https://drive.google.com/file/d/1Uaqc4QZGj8lkL2P8r41cl0OLhiY3k63q/view?usp=sharing) model.


4. Create folder `output/models` and put the .pth file inside.


5. To create the `answer.txt` file, run this command:
    ```
    python inference.py
    ```

## Results
The best accuracy I can get on testing data is around 0.8 using EfficientNet-B4.
| Model name         | Top 1 Accuracy on CodaLab  |
| :-: | :-: |
| [efficientnetb4_1.pth](https://drive.google.com/file/d/1Uaqc4QZGj8lkL2P8r41cl0OLhiY3k63q/view?usp=sharing)   |     0.801187         |
