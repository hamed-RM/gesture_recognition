import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
from keras import datasets, layers, models,activations
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.models import load_model
from sklearn.model_selection import train_test_split


def get_data(data_dir):
    data = []
    for label in classes:
        # count=0
        path = os.path.join(data_dir, label)
        class_num = classes.index(label)
        for img in os.listdir(path):

            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                # (thresh, im_bw)=cv2.threshold(img_arr,128,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
                # plt.imshow(im_bw)
                # plt.show()
                resized_arr = cv2.resize(img_arr, (50, 50))
                data.append([resized_arr, class_num])
                # count=count+1
            except Exception as e:
                print(e)
    return data
class cal_text_model():

    def __init__(self):
        pass

    def train_model(self,x_train,y_train,name):

        model=models.Sequential([layers.Conv2D(16, (2,2), activation='relu',input_shape=(50,50,1)),
                                    layers.MaxPooling2D((2, 2),strides=2,padding='same'),
                                    layers.Conv2D(32, (5, 5), activation='relu'),
                                    layers.MaxPooling2D((5, 5),strides=5,padding='same'),
                                    layers.Conv2D(64,(5,5),activation='relu'),
                                    layers.Flatten(),
                                    layers.Dense(128,activation='relu'),
                                    layers.Dropout(0.2),
                                    layers.Dense(37,activation=activations.softmax)
                                     ])
        model.compile(optimizer=keras.optimizers.SGD(lr=0.01,momentum=0.0,decay=0.0,nesterov=False),loss=keras.losses.SparseCategoricalCrossentropy(),metrics = ['accuracy'])
        model.summary()
        history=model.fit(x_train,y_train,epochs=20,batch_size=500)
        # print(history.history.keys())
        model.summary()
        acc=history.history['accuracy']
        # val_acc=history.history['val_accuracy']
        loss=history.history['loss']

        # val_loss=history.history['val_loss']

        epochs_range=range(20)


        fig, ax = plt.subplots()
        ax.plot(acc, color="blue")  # set line color to blue
        ax.set_xlabel('epoch')
        ax.set_ylabel('accuracy', color="blue")  # set y axis title to blue
        ax.tick_params(axis='y', colors="blue")  # set y axis tick labels to blue

        # We create another axis object. It shares the same x axis as ax, but the y-axis is separate.
        ax2 = ax.twinx()
        ax2.plot(loss, color="red")  # set line color to red
        ax2.set_ylabel('loss', color="red")  # set y axis title to red
        ax2.tick_params(axis='y', colors="red")
        plt.show()

        # predictions=model.predict_classes(x_valid)
        # predict_x=model.predict(x_valid)
        # predictions=np.argmax(predict_x,axis=1)
        # predictions=predictions.reshape(1,-1)[0]
        # print(classification_report(y_valid,predictions,target_names=classes))

        model.save(f'{name}_model.h5')
        # return model

datadir = 'data/cal_text_img/dcgan'
dirs=['data/cal_text_img/dcgan','data/cal_text_img/normal_img']
# classes = os.listdir(datadir)
classes=[str(num) for num in range(10)]
# print(classes)
# data=get_data('data/cal_text_img/dcgan')
data=get_data('data/cal_text_img/normal_img')
x_train=[]
x_valid=[]
y_train=[]
y_valid=[]


for d in dirs:
    data = get_data(d)

    x=[]
    y=[]
    for i,j in data:
        x.append(i)
        y.append(j)
    xt_train, xt_valid, yt_train, yt_valid = train_test_split(x, y, test_size=1/6, random_state=42)

    x_train.extend(np.array(xt_train))
    x_valid.extend(np.array(xt_valid))
    y_train.extend(np.array(yt_train))
    y_valid.extend(np.array(yt_valid))


x_train=np.array(x_train)
x_train.reshape(-1, 50, 50, 1)
y_train=np.array(y_train)

x_valid=np.array(x_valid)
x_valid.reshape(-1, 50, 50, 1)
y_valid=np.array(y_valid)
# print(y_valid.shape)
# print(x_valid.shape)
# breakpoint()
# print('training.....')
# m=cal_text_model()
# m.train_model(x_train, y_train,'normal')




#evaluation
n_sample_test = [500, 1000, 1500, 2000]
acc_dcgan = []
acc_normal=[]
#'dcgan+cnn model'
dcgan_model = load_model('dcgan_model.h5')
#'cnn model'
normal_model=load_model('normal_model.h5')
for n_test in n_sample_test:
    acc_dcgan.append(dcgan_model.evaluate(x_valid[:n_test], y_valid[:n_test])[1])
    acc_normal.append(normal_model.evaluate(x_valid[:n_test], y_valid[:n_test])[1])



x = [500, 1000, 1500, 2000]
plt.plot(x, acc_dcgan, label ='dcgan+cnn')
plt.plot(x, acc_normal, label ='cnn')
plt.xlabel("quantity")
plt.ylabel("accuracy")
plt.legend()
plt.show()
#


