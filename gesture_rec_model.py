import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
from keras import datasets, layers, models,activations
from keras.optimizers import RMSprop,Adam
from keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split


def get_data(data_dir):
    data = []
    for label in classes:
        path = os.path.join(data_dir, label)
        class_num = classes.index(label)
        for img in os.listdir(path):
            if(img=='desktop.ini'):
                continue
            try:
                img_arr = cv2.imread(os.path.join(path, img),cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (50, 50))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return data

def train_model(x_train,y_train,name):

    model=models.Sequential([layers.Conv2D(32, (5,5), activation='relu',input_shape=(50,50,1)),
                             layers.MaxPooling2D((2, 2),strides=2,padding='same'),
                             layers.Conv2D(64, (5, 5), activation=activations.sigmoid),
                             layers.MaxPooling2D((5, 5),strides=5,padding='same'),
                             layers.Flatten(),
                             layers.Dense(1024,activation='relu'),
                             layers.Dropout(0.6),
                             layers.Dense(10,activation=activations.softmax)
                             ])

    model.compile(optimizer=Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-08,decay=0.0),
                  loss=keras.losses.SparseCategoricalCrossentropy(),metrics = ['accuracy'])

    history=model.fit(x_train,y_train,epochs=10,batch_size=64)
    # print(history.history.keys())
    model.summary()
    acc=history.history['accuracy']
    loss=history.history['loss']

    epochs_range=range(10)

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


datadir='data/gesture_img'
classes=os.listdir(datadir)
data=get_data(data_dir=datadir)
x=[]
y=[]
for i,j in data:
    x.append(i)
    y.append(j)

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=1/6, random_state=42)


x_train=np.array(x_train)
x_valid=np.array(x_valid)

x_train.reshape(-1, 50, 50, 1)
y_train=np.array(y_train)

x_valid.reshape(-1, 50, 50, 1)
y_valid=np.array(y_valid)

train_model(x_train,y_train,'gesture')


gesture_model = load_model('gesture_model.h5')

predict_x=gesture_model.predict(x_valid)
predictions=np.argmax(predict_x,axis=1)
predictions=predictions.reshape(1,-1)[0]
# print(predictions)
print(classification_report(y_valid,predictions,target_names=classes))

