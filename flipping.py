from PIL import Image
import os

# 1. FLIPPING IMAGES
# files = os.listdir('data/training_images')
# print(len(files))

# count = 0
# for imageName in files:
#     im = Image.open("data/training_images/" + imageName)
#     # flip image
#     out = im.transpose(Image.FLIP_LEFT_RIGHT)
#     out.save('data/training_flipped/' + imageName.replace('.jpg', '_flip.jpg'))

#     count += 1
# print(f'Finish processing {count} images.')



# 2. CREATE NEW LABEL FILE
with open('data/training_augmented_labels.txt', 'w') as dst:
    with open('data/training_labels.txt', 'r') as src:
        count = 0
        for row in src:
            row = row.strip()
            imageName, label = row.split(' ')
            print(f'{imageName} {label}', file=dst)  # original one
            imageName = imageName.replace('.jpg', '_flip.jpg')
            print(f'{imageName} {label}', file=dst)  # flipped one
            
            count += 2
print(f'Finish processing {count} items.')
