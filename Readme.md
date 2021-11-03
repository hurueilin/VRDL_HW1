# VRDL HW1 - Bird Images Classification

## Environment
Hardware
* CPU: AMD Ryzen 5 3600 6-Core
* GPU: NVIDIA GeForce RTX 3070 8GB

Packages
* torch: 1.7.1+cu110
* torchvision: 0.8.2+cu110
* torch-summary: 1.4.4
* numpy: 1.19.5
* Center Loss
* EfficientNet

## Training

To train the model, run this command:

```
python Train.py
```

## Reproducing Submission
1. Clone this project or download ZIP.


2. Download the submitted .pth model.
https://drive.google.com/file/d/1Uaqc4QZGj8lkL2P8r41cl0OLhiY3k63q/view?usp=sharing

3. Create folder `output/models` and put the .pth file inside.


4. To create the `answer.txt` file, run this command:
    ```
    python inference.py
    ```

## Results
| Model name         | Top 1 Accuracy on CodaLab  |
| :-: | :-: |
| efficientnetb4_1.pth   |     0.801187         |
