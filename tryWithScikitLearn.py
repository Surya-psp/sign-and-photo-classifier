# import tensorflow as tf
# from tensorflow.python.framework.ops import EagerTensor
# from tensorflow.python.ops.resource_variable_ops import ResourceVariable
import numpy as np
# import pandas as pd
import copy
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
import PIL
import random
from sklearn import *
from sklearn import svm, metrics
import pickle

test_folder = r'D:\\SoftwareEngineering\\Extracted Faces\\Extracted Faces'
plt.figure(figsize=(20,20))
files = os.listdir(test_folder)
for i in range(10):
    files = random.choice(os.listdir(test_folder))
#     print(files)
    sub_path = os.path.join(test_folder, files)
    file = random.choice(os.listdir(sub_path))
    img_path = os.path.join(sub_path, file)
    # print(file)
    img = mpimg.imread(img_path)
    # print(img.shape)
    ax = plt.subplot(1, 10, i+1)
#     ax.title.set_text(file)
    plt.imshow(img)



def create_dataset(img_folder):
    IMG_HEIGHT = 128
    IMG_WIDTH = 128

    img_data_array = []
    class_name = []
    icount, scount = 0, 0
    for dir1 in os.listdir(img_folder):
        k = os.listdir(os.path.join(img_folder, dir1))
        for i in range(len(k)):
            if icount > 1000:
                icount = 0
                break
            if scount > 1000:
                break
            image_path = os.path.join(img_folder, dir1, k[i])
            if dir1 == 'train':
#                 scount += 1
                sub_path = os.path.join(img_folder, dir1, k[i])
                for img_file in os.listdir(os.path.join(img_folder, dir1, k[i])):
                    scount += 1
                    image_path = os.path.join(img_folder, dir1, k[i], img_file)
                    img = PIL.Image.open(r""+str(image_path))
                    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
                    img = np.array(img)
                    img.astype('float32')
        #             print(img)
                    img=img/255
                    img_data_array.append(img)
                    class_name.append(dir1)
            else:
                icount += 1
            img = PIL.Image.open(r""+str(image_path))
            img = img.resize((IMG_WIDTH, IMG_HEIGHT))
            img = np.array(img)
            img.astype('float32')
#             print(img)
            img=img/255
            img_data_array.append(img)
            class_name.append(dir1)
#             count += 1
    return img_data_array, class_name

IMG_FOLDER = r'D:\\SoftwareEngineering\\BinaryTraining'

img_data, class_name = create_dataset(IMG_FOLDER)

img_data_final = []
for i in img_data:
#     print(i.shape)
    i = i.reshape(i.shape[0]*i.shape[1]*i.shape[2], 1)
    img_data_final.append(i)


coupled_dataset = list(zip(class_name, img_data_final))
random.shuffle(coupled_dataset)

class_name, img_data_final = zip(*coupled_dataset)
# print(class_name)

class_name_final = []
for i in class_name:
    if i == 'Faces':
        class_name_final.append(0)
    else:
        class_name_final.append(1)
# print(class_name_final)

X_train = np.array(img_data_final[:1900])
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]))
# print(len(img_data_final))
print(X_train.shape)
Y_train = np.array(class_name_final[:1900]).reshape(1900,1)
Y_train.reshape([X_train.shape[0], 1])
print(Y_train.shape)
X_test = np.array(img_data_final[1901:2001])
Y_test = np.array(class_name_final[1901:2001])


# X_train.shape
X_train = X_train.reshape((1900, 49152))
# print(X_test.shape)
X_test = X_test.reshape((100, 49152))
Y_test = Y_test.reshape((100, 1))
Y_train = Y_train.reshape((1900,1))
# print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

clf = svm.SVC(gamma=0.0001)
clf.fit(X_train, Y_train)
predicted = clf.predict(X_test)

print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(Y_test, predicted)}\n"
)


disp = metrics.ConfusionMatrixDisplay.from_predictions(Y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()



IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_FOLDER = r'D:\\SoftwareEngineering\\BT\\Sign.png'
img = PIL.Image.open(r""+str(IMG_FOLDER))
img = img.resize((IMG_WIDTH, IMG_HEIGHT))
img = np.array(img)
img.astype('float32')
#             print(img)
img=img/255
img = img.reshape(1, img.shape[0]*img.shape[1]*img.shape[2])

predicted = clf.predict(img)
print(predicted)

print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report([1], predicted)}\n"
)

print(clf.coef0)

pickle.dump(clf, open(r'D:\\SoftwareEngineering\\model.sav', 'wb'))
model1 = pickle.load(open(r'D:\\SoftwareEngineering\\model.sav', 'rb'))

predicted1 = model1.predict(img)
print(predicted1)