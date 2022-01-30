
from matplotlib.pyplot import imshow
import matplotlib.cm as cm
import matplotlib.pylab as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import PIL
import tensorflow as tf
from PIL import ImageFilter
import cv2
import itertools
import random
from tensorflow import keras 
import imutils
from imutils import paths
import os
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D , UpSampling2D ,Conv2DTranspose
from tensorflow.keras import backend as K


def pil_image(img_path):
    pil_im =PIL.Image.open(img_path).convert('L')
    pil_im=pil_im.resize((105,105))
    #imshow(np.asarray(pil_im))
    return pil_im


# # Augumentation Steps 
# 1) Noise
# 2) Blur
# 3) Perpective Rotation
# 4) Shading
# 5) Variable Character Spacing
# 6) Variable Aspect Ratio



def noise_image(pil_im):
    # Adding Noise to image
    img_array = np.asarray(pil_im)
    mean = 0.0   # some constant
    std = 5   # some constant (standard deviation)
    noisy_img = img_array + np.random.normal(mean, std, img_array.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255)
    noise_img = PIL.Image.fromarray(np.uint8(noisy_img_clipped)) # output
    #imshow((noisy_img_clipped ).astype(np.uint8))
    noise_img=noise_img.resize((105,105))
    return noise_img

def blur_image(pil_im):
    #Adding Blur to image 
    blur_img = pil_im.filter(ImageFilter.GaussianBlur(radius=3)) # ouput
    #imshow(blur_img)
    blur_img=blur_img.resize((105,105))
    return blur_img


def affine_rotation(img):
    
    #img=cv2.imread(img_path,0)
    rows, columns = img.shape

    point1 = np.float32([[10, 10], [30, 10], [10, 30]])
    point2 = np.float32([[20, 15], [40, 10], [20, 40]])

    A = cv2.getAffineTransform(point1, point2)

    output = cv2.warpAffine(img, A, (columns, rows))
    affine_img = PIL.Image.fromarray(np.uint8(output)) # affine rotated output
    #imshow(output)
    affine_img=affine_img.resize((105,105))
    return affine_img
   


def gradient_fill(image):
    #image=cv2.imread(img_path,0)
    laplacian = cv2.Laplacian(image,cv2.CV_64F)
    laplacian = cv2.resize(laplacian, (105, 105))
    return laplacian


# ## Preparing Dataset

data_path = "TextRecognitionDataGenerator/font_patch2/"
data=[]
labels=[]
imagePaths = sorted(list(paths.list_images(data_path)))
random.seed(42)
random.shuffle(imagePaths)
num_labelss=7
def conv_label(label):
    if   label == 'kakugo_0':
        return 0
    elif label == 'kakugo_2':
        return 1
    elif label == 'kakugo_4':
        return 2
    elif label == 'kakugo_6':
        return 3
    elif label == 'kakugo_8':
        return 4
    elif label == 'marugo_4':
        return 5
    elif label == 'mincho_4':
        return 6    

augument=["blur","noise","affine","gradient"]
"""a=itertools.combinations(augument, 4)

for i in list(a): 
    print(list(i))"""

##preprocess for argument
counter=0
for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    label = conv_label(label)
    pil_img = pil_image(imagePath)
    #imshow(pil_img)
    
    # Adding original image
    org_img = img_to_array(pil_img)
    #print(org_img.shape)
    data.append(org_img)
    labels.append(label)
    
    #augument=["noise","blur","affine","gradient"]
    for l in range(0,len(augument)):
    
        a=itertools.combinations(augument, l+1)

        for i in list(a): 
            combinations=list(i)
            #print(len(combinations))
            temp_img = pil_img
            for j in combinations:
            
                if j == 'noise':
                    # Adding Noise image
                    temp_img = noise_image(temp_img)
                    
                elif j == 'blur':
                    # Adding Blur image
                    temp_img = blur_image(temp_img)
                    #imshow(blur_img)
                    
    
                elif j == 'affine':
                    open_cv_affine = np.array(pil_img)
                    # Adding affine rotation image
                    temp_img = affine_rotation(open_cv_affine)

                elif j == 'gradient':
                    open_cv_gradient = np.array(pil_img)
                    # Adding gradient image
                    temp_img = gradient_fill(open_cv_gradient)
  
            temp_img = img_to_array(temp_img)
            data.append(temp_img)
            labels.append(label)


data = np.asarray(data, dtype="float") / 255.0
labels = np.array(labels)
print("Success")
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42,stratify=labels)


# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=num_labelss)
testY = to_categorical(testY, num_classes=num_labelss)

### load and evaluate
from tensorflow.keras.models import load_model
model = load_model('hiragana_128_100_adam_20220110-055359.h5')
score = model.evaluate(testX, testY, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

img_path="TextRecognitionDataGenerator/sample/あけまして おめでとう_2.jpg"
pil_im =PIL.Image.open(img_path).convert('L')
pil_im=blur_image(pil_im)
org_img = img_to_array(pil_im)


def rev_conv_label(label):
    if   label == 0:
        return 'kakugo_0'
    elif label == 1:
        return 'kakugo_2'
    elif label == 2:
        return 'kakugo_4'
    elif label == 3:
        return 'kakugo_6'
    elif label == 4:
        return 'kakugo_8'
    elif label == 5:
        return 'marugo_4'
    elif label == 6:
        return 'mincho_4' 

###3##33f3gaa##sasdfasaadata=[]
data.append(org_img)
data = np.asarray(data, dtype="float") / 255.0


y = model.predict_classes(data)



label = rev_conv_label(int(y[0]))
fig, ax = plt.subplots(1)
ax.imshow(pil_im, interpolation='nearest', cmap=cm.gray)
ax.text(5, 5, label , bbox={'facecolor': 'white', 'pad': 10})
plt.show()

