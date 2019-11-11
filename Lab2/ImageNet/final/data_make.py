import cv2
import os
import numpy as np

path = r'C:\Users\shubham\Downloads\HIDL\Lab2\DLLabs\Lab2\ImageNet\minc-2500-tiny'

test_path = path + "/test"
train_path = path + "\\train"
val_path = path + "\\val"

label_list = os.listdir(train_path)
label_indices_mapping = {}

index = 0
for x in label_list:
    label_indices_mapping[x] = index
    index += 1

print(label_indices_mapping)

train_data = []
train_label = []

for x in label_list:
    _final_path = train_path + '\\' + x
    for y in os.listdir(_final_path):
        _image_path = _final_path + '\\' + y
        im = cv2.imread(_image_path)
        train_data.append(im)
        train_label.append(label_indices_mapping.get(x))

print(len(train_data))

for x in label_list:
    _final_path = val_path + '\\' + x
    for y in os.listdir(_final_path):
        _image_path = _final_path + '\\' + y
        im = cv2.imread(_image_path)
        train_data.append(im)
        train_label.append(label_indices_mapping.get(x))

test_data = []
test_label = []

for x in label_list:
    _final_path = test_path + '\\' + x
    for y in os.listdir(_final_path):
        _image_path = _final_path + '\\' + y
        im = cv2.imread(_image_path)
        test_data.append(im)
        test_label.append(label_indices_mapping.get(x))

train_data = np.asarray(train_data)
train_label = np.asarray(train_label)

test_data = np.asarray(test_data)
test_label = np.asarray(test_label)

print(train_data.shape)

np.save('train_data.npy', train_data)
np.save('train_label.npy', train_label)
np.save('test_data.npy', test_data)
np.save('test_label.npy', test_label)
