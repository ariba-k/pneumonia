from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, Dense, AveragePooling2D, BatchNormalizationV2, Dropout, MaxPool2D, Softmax, Flatten
from tensorflow.python.keras import Model, Sequential, Input
from tensorflow.python.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.python.keras.applications import Xception, VGG16, VGG19, InceptionV3, InceptionResNetV2, MobileNet, MobileNetV2, DenseNet121, NASNetLarge
from PIL import Image
from tensorflow.python.keras.preprocessing.image import image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os.path
from glob import glob
from itertools import chain
from skimage.transform import resize
from  skimage.io import imsave
import cv2
import random



def onehot(indices, length):
    a = np.array([0] * length)
    a[indices] = 1
    return a


# load labels
labels_df = pd.read_csv('/home/lambda/Desktop/pneumonia/cxr14/Data.csv')
label_col = labels_df['Finding Labels']
img_name_col = labels_df['Image Index']
print(label_col)
unique_labels = set()
for l in label_col:
    unique_labels.update(l.split('|'))

unique_labels = list(unique_labels)
unique_labels.remove('No Finding')
unique_labels.insert(0, 'No Finding')
label_map = {unique_labels[i]: i for i in range(len(unique_labels))}

onehot_labels = []
for r in label_col:
    onehot_labels.append(onehot([label_map[s] for s in r.split('|')], len(unique_labels)))
onehot_labels = np.array(onehot_labels)


#aug_imgs = []
#aug_labels = []
#print(imgs)
#wildcard_aug = '/home/lambda/Desktop/Chest_XRay14_images_002/images/*.png'
#path_to_aug_imgs = glob(wildcard_aug)

img_to_onehot = {name: label for name, label in zip(img_name_col, onehot_labels)}
imgs = []
labels = []
wildcard = '/home/lambda/Desktop/pneumonia/cxr14/Chest_XRay14_images_002/images/*.png'
path_to_imgs = glob(wildcard)
for path_to_img in path_to_imgs:
    file_name = os.path.basename(path_to_img)
    if file_name in img_to_onehot:
        imgs.append(path_to_img)
        labels.append(img_to_onehot[file_name])

'''for path_to_aug_img in path_to_aug_imgs:
    file_name = os.path.basename(path_to_aug_img)
    if file_name in img_to_onehot:
        aug_imgs.append(path_to_aug_img)
        aug_labels.append(img_to_onehot[file_name])'''

class DataGen:
    def __init__(self, path_to_imgs,labels):
        self.path_to_imgs = path_to_imgs
        self.labels = labels


    def __call__(self):
        for path_to_img, label in zip(path_to_imgs, labels):
            img = plt.imread(path_to_img)
            img = resize(img, img_dim)
            img = norm_image(img)
            if len(img.shape) == 3:
                img = img[:, :, 0]
            yield img, label

'''class AugDataGen:
    def __init__(self,path_to_aug_imgs,aug_labels):
        self.path_to_aug_imgs = path_to_aug_imgs
        self.aug_labels = aug_labels
    def __call__(self):
        for path_to_aug_img, aug_label in zip(path_to_aug_imgs, aug_labels):
            img_to_transform = plt.imread(path_to_aug_img)
            transformations = {
                'rotate': rotate_image,
                'flip': flip_image,
                'stretch': stretch_image
            }
            num_trans_available = random.randint(1, len(transformations))
            for _ in range(0, num_trans_available):
                key = random.choice(list(transformations))
                transformed_img = transformations[key](img_to_transform)
            new_path = os.path.join(path, 'augmented_images')
            imsave(new_path, transformed_img)
            yield transformed_img, aug_label'''''


path = '/home/lambda/Desktop/Chest_XRay14_images_002/images'

def rotate_image(img, angle=random.randint(0,30)):
    row, col, chan = img.shape
    center = tuple(np.array([row, col])/2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rot_img = cv2.warpAffine(img, rot_mat, (col, row))
    return rot_img

def flip_image(img):
    img = np.expand_dims(img)
    tf_img = tf.convert_to_tensor(img) #requires 3D input
    flip_img = tf.image.random_flip_left_right(tf_img)
    return flip_img.numpy() #returns an array

def stretch_image(img):
    width = random.randint(0, int(len(img) * (1 / 3)))
    height = int(width / 2)
    stretch_img = cv2.resize(np.float32(img), (width, height))
    return stretch_img

def norm_image(img):
    mean = img.mean()
    std = img.std()
    pixels = (img - mean)/std
    return pixels


img_train, img_valid, label_train, label_valid = train_test_split(imgs, labels, test_size=0.2, shuffle=True)

batch_size = 16
epochs = 50
lab_dim = len(unique_labels)
img_dim = (512, 512)
lr = 1e-3


#data_gen_train = DataGen(img_train, label_train)
#aug_data_gen_train = AugDataGen(aug_imgs, aug_labels)
datagen_train = DataGen(img_train, label_train)
datagen_valid = DataGen(img_valid, label_valid)
ds_img_train = tf.data.Dataset.from_generator(datagen_train, (tf.float32, tf.float32), (tf.TensorShape(img_dim), tf.TensorShape(lab_dim)))\
         .map(lambda img, lab: (tf.expand_dims(img, axis=-1), lab))\
         .map(lambda img, lab: (tf.tile(img, tf.constant([1, 1, 3])), lab))\
         .shuffle(128)\
         .batch(batch_size, drop_remainder=True)

ds_img_valid = tf.data.Dataset.from_generator(datagen_valid, (tf.float32, tf.float32), (tf.TensorShape(img_dim), tf.TensorShape(lab_dim)))\
         .map(lambda img, lab: (tf.expand_dims(img, axis=-1), lab))\
         .map(lambda img, lab: (tf.tile(img, tf.constant([1, 1, 3])), lab))\
         .shuffle(128)\
         .batch(batch_size, drop_remainder=True)

#RESNET50
num_classes = 15
input_shape = img_dim
input_shape += (3,)
#valid = ds_img_valid.map(augment_image)
#train = ds_img_train.map(augment_image)

base_model_xception = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
x1 = Flatten()(base_model_xception.get_output_at(-1))
x1 = Dense(1024, activation='relu')(x1)
output1 = Dense(num_classes, activation='sigmoid')(x1)
input1 = base_model_xception.input

base_model_inception = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
x2 = Flatten()(base_model_inception.get_output_at(-1))
x2 = Dense(1024, activation='relu')(x2)
output2 = Dense(num_classes, activation='sigmoid')(x2)
input2 = base_model_inception.input

model = Model([input1, input2], [output1, output2])
model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr),
              loss=tf.nn.sigmoid_cross_entropy_with_logits, metrics=['accuracy'])
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=lr)
file_name = 'weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5'
save_model = tf.keras.callbacks.ModelCheckpoint(
    '/home/lambda/Desktop/Chest_XRay14_images_002/Saved_Models/{}'.format(file_name), monitor='val_loss')
train_history = model.fit(ds_img_train, epochs=epochs, callbacks=[reduce_lr, early_stopping, save_model],
                          validation_data=ds_img_valid)


#Loss Training History
plt.figure(figsize=(9, 9))
loss = train_history.history['loss']
val_loss = train_history.history['val_loss']
plt.subplot(2, 1, 1)
plt.plot(loss)
plt.plot(val_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc='best')
plt.show()

#Accuracy Training History
acc = train_history.history['accuracy']
val_acc = train_history.history['val_accuracy']
plt.subplot(2, 1, 2)
plt.plot(acc)
plt.plot(val_acc)
plt.legend(['train', 'test'], loc='best')
plt.show()

