---
layout: post
title: "Range of Convolutional Neural Networks on Fashion-MNIST dataset"
date: 2017-09-29 03:57
comments: true
categories: 
- Deep Learning
- Convolutional Neural Networks
---


Fashion MNIST is a drop-in replacement for the very well known, machine learning hello world, MNIST dataset. It has same number of training and test examples and the images have the same 28x28 size and there are a total of 10 classes/labels, you can read more about the dataset here : [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)


In this post we will be trying out different models and compare their results:

## List of models

1. 2 Layer Neural Netwoek
2. CNN with 1 Convolutional Layer
3. CNN with 3 Convolutional Layers
4. VGG Like Model
5. VGG Like Model With Batchnorm

<!-- more -->

## Approach

I split the original training data into 80% training and 20% validation. This helps to see weather we're over-fitting on the training data and weather we should lower the learning rate and train for more epochs if validation accuracy is higher than training accuracy or stop over-training if training accuracy shift higher than the validation.

To be consistent here, all the models are initially trained for 10 epochs and another 10 epochs with a lower learning late. After the initial 20 epochs, I added data augmentation, which generates new training samples by rotating, shifting and zooming on the training samples, and trained for another 50 epochs.

Also, to avoid hot encoding the labels, I decided to use `sparse_categorical_crossentropy` when compiling the models.

## Observations
All the models achieved a higher accuracy after using data augmentation. Almost always use data augmentation!!

VGG Like Model With Batchnorm performed the best and achieved a accuarcy of 94% using data augmentation.


## Fun Fact

If you uncomment the code in **Drop-in Replacement you said?** section, you'll be able to run all the models on MNIST instead of Fashion-MNIST.
It is much easier to get +99.5% results on MNIST. However, as you can see by running the models on both datasets, it gets relatively harder to squeeze accuracy on the Fashion-MNIST dataset. 


## Required Libaries


```python
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
np.random.seed(12345)
%matplotlib inline
```

    Using TensorFlow backend.


## Download and Load Fashion-MNIST


```python
batch_size = 512
```


```python
train_images_path = keras.utils.get_file('train-images-idx3-ubyte.gz', 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-images-idx3-ubyte.gz')
train_labels_path = keras.utils.get_file('train-labels-idx1-ubyte.gz', 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-labels-idx1-ubyte.gz')
test_images_path = keras.utils.get_file('t10k-images-idx3-ubyte.gz', 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-images-idx3-ubyte.gz')
test_labels_path = keras.utils.get_file('t10k-labels-idx1-ubyte.gz', 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-labels-idx1-ubyte.gz')
```


```python
def load_mnist(images_path, labels_path):
    import os
    import gzip
    import numpy as np

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

```


```python
X_train_orig, y_train_orig = load_mnist(train_images_path, train_labels_path)
X_test, y_test = load_mnist(test_images_path, test_labels_path)
X_train_orig = X_train_orig.astype('float32')
X_test = X_test.astype('float32')
X_train_orig /= 255
X_test /= 255
```

## Drop-in Replacement you said?
As I said at the beginning, fashion MNIST is drop-in replacement for MNINT. In case you want to run all these models on MNIST and compare the results. Uncomment the next section and everything should work automatically.


```python
# from keras.datasets import mnist
# (X_train_orig, y_train_orig), (X_test, y_test) = mnist.load_data()
# X_train_orig = X_train_orig.reshape(60000, 784)
# X_test = X_test.reshape(10000, 784)
# X_train_orig = X_train_orig.astype('float32')
# X_test = X_test.astype('float32')
# X_train_orig /= 255
# X_test /= 255
```


```python
print(X_train_orig.shape)
print(y_train_orig.shape)
print(X_test.shape)
print(y_test.shape)
```

    (60000, 784)
    (60000,)
    (10000, 784)
    (10000,)



```python
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train_orig, y_train_orig, test_size=0.2, random_state=12345)
```


```python
print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
```

    (48000, 784)
    (48000,)
    (12000, 784)
    (12000,)



```python
plt.imshow(X_train[1, :].reshape((28, 28)))
```




    <matplotlib.image.AxesImage at 0x7febae3da908>



{% img /images/output_13_1.png %}

## 2 Layer Neural Network


```python
model = Sequential([
    Dense(512, input_shape=(784,), activation='relu'),
    Dense(128, activation = 'relu'),
    Dense(10, activation='softmax')
])
```


```python
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 512)               401920    
    _________________________________________________________________
    dense_2 (Dense)              (None, 128)               65664     
    _________________________________________________________________
    dense_3 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 468,874
    Trainable params: 468,874
    Non-trainable params: 0
    _________________________________________________________________



```python
model.compile(optimizer=Adam(lr=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```


```python
history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=20,
                    verbose=1,
                    validation_data=(X_val, y_val))
```

    Train on 48000 samples, validate on 12000 samples
    Epoch 1/20
    48000/48000 [==============================] - 1s - loss: 0.6354 - acc: 0.7800 - val_loss: 0.4209 - val_acc: 0.8569
    Epoch 2/20
    48000/48000 [==============================] - 0s - loss: 0.4120 - acc: 0.8564 - val_loss: 0.3947 - val_acc: 0.8657
    Epoch 3/20
    48000/48000 [==============================] - 0s - loss: 0.3717 - acc: 0.8675 - val_loss: 0.3657 - val_acc: 0.8750
    Epoch 4/20
    48000/48000 [==============================] - 0s - loss: 0.3400 - acc: 0.8776 - val_loss: 0.3263 - val_acc: 0.8866
    Epoch 5/20
    48000/48000 [==============================] - 0s - loss: 0.3181 - acc: 0.8855 - val_loss: 0.3122 - val_acc: 0.8871
    Epoch 6/20
    48000/48000 [==============================] - 0s - loss: 0.2965 - acc: 0.8933 - val_loss: 0.3192 - val_acc: 0.8876
    Epoch 7/20
    48000/48000 [==============================] - 0s - loss: 0.2855 - acc: 0.8953 - val_loss: 0.3082 - val_acc: 0.8907
    Epoch 8/20
    48000/48000 [==============================] - 0s - loss: 0.2728 - acc: 0.8992 - val_loss: 0.2893 - val_acc: 0.8978
    Epoch 9/20
    48000/48000 [==============================] - 0s - loss: 0.2608 - acc: 0.9052 - val_loss: 0.3087 - val_acc: 0.8871
    Epoch 10/20
    48000/48000 [==============================] - 0s - loss: 0.2501 - acc: 0.9067 - val_loss: 0.2865 - val_acc: 0.8967
    Epoch 11/20
    48000/48000 [==============================] - 0s - loss: 0.2392 - acc: 0.9117 - val_loss: 0.2930 - val_acc: 0.8967
    Epoch 12/20
    48000/48000 [==============================] - 0s - loss: 0.2289 - acc: 0.9161 - val_loss: 0.2985 - val_acc: 0.8953
    Epoch 13/20
    48000/48000 [==============================] - 0s - loss: 0.2251 - acc: 0.9173 - val_loss: 0.2922 - val_acc: 0.8960
    Epoch 14/20
    48000/48000 [==============================] - 0s - loss: 0.2124 - acc: 0.9214 - val_loss: 0.2962 - val_acc: 0.8964
    Epoch 15/20
    48000/48000 [==============================] - 0s - loss: 0.2017 - acc: 0.9253 - val_loss: 0.2751 - val_acc: 0.9038
    Epoch 16/20
    48000/48000 [==============================] - 0s - loss: 0.1966 - acc: 0.9270 - val_loss: 0.2858 - val_acc: 0.9011
    Epoch 17/20
    48000/48000 [==============================] - 0s - loss: 0.1874 - acc: 0.9309 - val_loss: 0.2918 - val_acc: 0.8989
    Epoch 18/20
    48000/48000 [==============================] - 0s - loss: 0.1841 - acc: 0.9312 - val_loss: 0.2920 - val_acc: 0.8984
    Epoch 19/20
    48000/48000 [==============================] - 0s - loss: 0.1812 - acc: 0.9338 - val_loss: 0.2831 - val_acc: 0.9004
    Epoch 20/20
    48000/48000 [==============================] - 0s - loss: 0.1673 - acc: 0.9381 - val_loss: 0.2984 - val_acc: 0.9013



```python
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

    Test loss: 0.357285452712
    Test accuracy: 0.8879


## CNN with 1 Convolutional Layer


```python
img_rows = 28
img_cols = 28
input_shape = (img_rows, img_cols, 1)
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)
```


```python
cnn1 = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```


```python
cnn1.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])
```


```python
cnn1.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=10,
          verbose=1,
          validation_data=(X_val, y_val))
```

    Train on 48000 samples, validate on 12000 samples
    Epoch 1/10
    48000/48000 [==============================] - 1s - loss: 0.6535 - acc: 0.7764 - val_loss: 0.4212 - val_acc: 0.8563
    Epoch 2/10
    48000/48000 [==============================] - 0s - loss: 0.4004 - acc: 0.8595 - val_loss: 0.3474 - val_acc: 0.8813
    Epoch 3/10
    48000/48000 [==============================] - 0s - loss: 0.3477 - acc: 0.8769 - val_loss: 0.3211 - val_acc: 0.8893
    Epoch 4/10
    48000/48000 [==============================] - 0s - loss: 0.3228 - acc: 0.8848 - val_loss: 0.2988 - val_acc: 0.8969
    Epoch 5/10
    48000/48000 [==============================] - 0s - loss: 0.2998 - acc: 0.8940 - val_loss: 0.2789 - val_acc: 0.9033
    Epoch 6/10
    48000/48000 [==============================] - 0s - loss: 0.2865 - acc: 0.8975 - val_loss: 0.2782 - val_acc: 0.9018
    Epoch 7/10
    48000/48000 [==============================] - 0s - loss: 0.2721 - acc: 0.9030 - val_loss: 0.2709 - val_acc: 0.9053
    Epoch 8/10
    48000/48000 [==============================] - 0s - loss: 0.2654 - acc: 0.9036 - val_loss: 0.2531 - val_acc: 0.9102
    Epoch 9/10
    48000/48000 [==============================] - 0s - loss: 0.2534 - acc: 0.9083 - val_loss: 0.2538 - val_acc: 0.9063
    Epoch 10/10
    48000/48000 [==============================] - 0s - loss: 0.2481 - acc: 0.9094 - val_loss: 0.2823 - val_acc: 0.8995









```python
cnn1.optimizer.lr = 0.0001
```


```python
cnn1.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=10,
          verbose=1,
          validation_data=(X_val, y_val))
```

    Train on 48000 samples, validate on 12000 samples
    Epoch 1/10
    48000/48000 [==============================] - 0s - loss: 0.2361 - acc: 0.9138 - val_loss: 0.2467 - val_acc: 0.9130
    Epoch 2/10
    48000/48000 [==============================] - 0s - loss: 0.2254 - acc: 0.9188 - val_loss: 0.2436 - val_acc: 0.9139
    Epoch 3/10
    48000/48000 [==============================] - 0s - loss: 0.2203 - acc: 0.9195 - val_loss: 0.2362 - val_acc: 0.9177
    Epoch 4/10
    48000/48000 [==============================] - 0s - loss: 0.2104 - acc: 0.9228 - val_loss: 0.2366 - val_acc: 0.9167
    Epoch 5/10
    48000/48000 [==============================] - 0s - loss: 0.2070 - acc: 0.9238 - val_loss: 0.2276 - val_acc: 0.9187
    Epoch 6/10
    48000/48000 [==============================] - 0s - loss: 0.1971 - acc: 0.9278 - val_loss: 0.2254 - val_acc: 0.9229
    Epoch 7/10
    48000/48000 [==============================] - 0s - loss: 0.1913 - acc: 0.9301 - val_loss: 0.2348 - val_acc: 0.9149
    Epoch 8/10
    48000/48000 [==============================] - 0s - loss: 0.1860 - acc: 0.9313 - val_loss: 0.2271 - val_acc: 0.9194
    Epoch 9/10
    48000/48000 [==============================] - 0s - loss: 0.1809 - acc: 0.9344 - val_loss: 0.2220 - val_acc: 0.9223
    Epoch 10/10
    48000/48000 [==============================] - 0s - loss: 0.1735 - acc: 0.9368 - val_loss: 0.2174 - val_acc: 0.9237









```python
score = cnn1.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

    Test loss: 0.250880071822
    Test accuracy: 0.9123


#### Data Augmentation


```python
gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                               height_shift_range=0.08, zoom_range=0.08)
batches = gen.flow(X_train, y_train, batch_size=batch_size)
val_batches = gen.flow(X_val, y_val, batch_size=batch_size)
```


```python
cnn1.fit_generator(batches, steps_per_epoch=48000//batch_size, epochs=50, 
                    validation_data=val_batches, validation_steps=12000//batch_size, use_multiprocessing=True)
```

    Epoch 1/50
    93/93 [==============================] - 7s - loss: 0.5429 - acc: 0.7988 - val_loss: 0.4489 - val_acc: 0.8303
    Epoch 2/50
    93/93 [==============================] - 6s - loss: 0.4493 - acc: 0.8327 - val_loss: 0.4163 - val_acc: 0.8443
    Epoch 3/50
    93/93 [==============================] - 6s - loss: 0.4383 - acc: 0.8365 - val_loss: 0.3987 - val_acc: 0.8533
    Epoch 4/50
    93/93 [==============================] - 6s - loss: 0.4167 - acc: 0.8440 - val_loss: 0.3855 - val_acc: 0.8594
    Epoch 5/50
    93/93 [==============================] - 6s - loss: 0.4039 - acc: 0.8486 - val_loss: 0.3835 - val_acc: 0.8585
    Epoch 6/50
    93/93 [==============================] - 6s - loss: 0.4013 - acc: 0.8498 - val_loss: 0.3762 - val_acc: 0.8633
    Epoch 7/50
    93/93 [==============================] - 6s - loss: 0.3855 - acc: 0.8555 - val_loss: 0.3643 - val_acc: 0.8633
    Epoch 8/50
    93/93 [==============================] - 6s - loss: 0.3817 - acc: 0.8575 - val_loss: 0.3545 - val_acc: 0.8700
    Epoch 9/50
    93/93 [==============================] - 6s - loss: 0.3739 - acc: 0.8602 - val_loss: 0.3565 - val_acc: 0.8693
    Epoch 10/50
    93/93 [==============================] - 6s - loss: 0.3698 - acc: 0.8611 - val_loss: 0.3494 - val_acc: 0.8706
    Epoch 11/50
    93/93 [==============================] - 6s - loss: 0.3654 - acc: 0.8619 - val_loss: 0.3487 - val_acc: 0.8697
    Epoch 12/50
    93/93 [==============================] - 6s - loss: 0.3572 - acc: 0.8653 - val_loss: 0.3448 - val_acc: 0.8737
    Epoch 13/50
    93/93 [==============================] - 6s - loss: 0.3552 - acc: 0.8671 - val_loss: 0.3333 - val_acc: 0.8791
    Epoch 14/50
    93/93 [==============================] - 6s - loss: 0.3552 - acc: 0.8667 - val_loss: 0.3485 - val_acc: 0.8702
    Epoch 15/50
    93/93 [==============================] - 6s - loss: 0.3513 - acc: 0.8693 - val_loss: 0.3348 - val_acc: 0.8756
    Epoch 16/50
    93/93 [==============================] - 6s - loss: 0.3461 - acc: 0.8700 - val_loss: 0.3273 - val_acc: 0.8807
    Epoch 17/50
    93/93 [==============================] - 6s - loss: 0.3456 - acc: 0.8706 - val_loss: 0.3333 - val_acc: 0.8774
    Epoch 18/50
    93/93 [==============================] - 6s - loss: 0.3386 - acc: 0.8745 - val_loss: 0.3257 - val_acc: 0.8785
    Epoch 19/50
    93/93 [==============================] - 6s - loss: 0.3349 - acc: 0.8761 - val_loss: 0.3180 - val_acc: 0.8811
    Epoch 20/50
    93/93 [==============================] - 6s - loss: 0.3331 - acc: 0.8757 - val_loss: 0.3175 - val_acc: 0.8848
    Epoch 21/50
    93/93 [==============================] - 6s - loss: 0.3321 - acc: 0.8757 - val_loss: 0.3268 - val_acc: 0.8753
    Epoch 22/50
    93/93 [==============================] - 6s - loss: 0.3303 - acc: 0.8770 - val_loss: 0.3149 - val_acc: 0.8836
    Epoch 23/50
    93/93 [==============================] - 6s - loss: 0.3238 - acc: 0.8791 - val_loss: 0.3114 - val_acc: 0.8845
    Epoch 24/50
    93/93 [==============================] - 6s - loss: 0.3235 - acc: 0.8791 - val_loss: 0.3020 - val_acc: 0.8903
    Epoch 25/50
    93/93 [==============================] - 6s - loss: 0.3251 - acc: 0.8783 - val_loss: 0.3100 - val_acc: 0.8843
    Epoch 26/50
    93/93 [==============================] - 6s - loss: 0.3219 - acc: 0.8817 - val_loss: 0.3054 - val_acc: 0.8884
    Epoch 27/50
    93/93 [==============================] - 6s - loss: 0.3183 - acc: 0.8805 - val_loss: 0.3007 - val_acc: 0.8931
    Epoch 28/50
    93/93 [==============================] - 6s - loss: 0.3146 - acc: 0.8823 - val_loss: 0.3076 - val_acc: 0.8871
    Epoch 29/50
    93/93 [==============================] - 6s - loss: 0.3089 - acc: 0.8833 - val_loss: 0.3043 - val_acc: 0.8881
    Epoch 30/50
    93/93 [==============================] - 6s - loss: 0.3140 - acc: 0.8841 - val_loss: 0.3013 - val_acc: 0.8866
    Epoch 31/50
    93/93 [==============================] - 6s - loss: 0.3087 - acc: 0.8843 - val_loss: 0.2933 - val_acc: 0.8901
    Epoch 32/50
    93/93 [==============================] - 6s - loss: 0.3108 - acc: 0.8820 - val_loss: 0.2974 - val_acc: 0.8953
    Epoch 33/50
    93/93 [==============================] - 6s - loss: 0.3064 - acc: 0.8871 - val_loss: 0.3004 - val_acc: 0.8903
    Epoch 34/50
    93/93 [==============================] - 6s - loss: 0.3055 - acc: 0.8859 - val_loss: 0.2916 - val_acc: 0.8930
    Epoch 35/50
    93/93 [==============================] - 6s - loss: 0.3047 - acc: 0.8862 - val_loss: 0.3002 - val_acc: 0.8890
    Epoch 36/50
    93/93 [==============================] - 6s - loss: 0.3006 - acc: 0.8880 - val_loss: 0.2881 - val_acc: 0.8953
    Epoch 37/50
    93/93 [==============================] - 6s - loss: 0.3063 - acc: 0.8856 - val_loss: 0.3006 - val_acc: 0.8888
    Epoch 38/50
    93/93 [==============================] - 6s - loss: 0.2984 - acc: 0.8874 - val_loss: 0.3068 - val_acc: 0.8862
    Epoch 39/50
    93/93 [==============================] - 6s - loss: 0.3032 - acc: 0.8859 - val_loss: 0.2894 - val_acc: 0.8939
    Epoch 40/50
    93/93 [==============================] - 6s - loss: 0.2996 - acc: 0.8883 - val_loss: 0.3023 - val_acc: 0.8871
    Epoch 41/50
    93/93 [==============================] - 6s - loss: 0.3003 - acc: 0.8876 - val_loss: 0.3014 - val_acc: 0.8899
    Epoch 42/50
    93/93 [==============================] - 6s - loss: 0.2933 - acc: 0.8894 - val_loss: 0.2886 - val_acc: 0.8928
    Epoch 43/50
    93/93 [==============================] - 6s - loss: 0.2939 - acc: 0.8890 - val_loss: 0.3088 - val_acc: 0.8851
    Epoch 44/50
    93/93 [==============================] - 6s - loss: 0.2952 - acc: 0.8878 - val_loss: 0.2854 - val_acc: 0.8960
    Epoch 45/50
    93/93 [==============================] - 6s - loss: 0.2923 - acc: 0.8910 - val_loss: 0.2846 - val_acc: 0.8964
    Epoch 46/50
    93/93 [==============================] - 6s - loss: 0.2891 - acc: 0.8917 - val_loss: 0.2928 - val_acc: 0.8920
    Epoch 47/50
    93/93 [==============================] - 6s - loss: 0.2903 - acc: 0.8898 - val_loss: 0.2829 - val_acc: 0.8981
    Epoch 48/50
    93/93 [==============================] - 6s - loss: 0.2877 - acc: 0.8917 - val_loss: 0.2863 - val_acc: 0.8967
    Epoch 49/50
    93/93 [==============================] - 6s - loss: 0.2903 - acc: 0.8896 - val_loss: 0.2831 - val_acc: 0.8978
    Epoch 50/50
    93/93 [==============================] - 6s - loss: 0.2848 - acc: 0.8930 - val_loss: 0.2858 - val_acc: 0.8968









```python
score = cnn1.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

    Test loss: 0.23108966648
    Test accuracy: 0.9153


### CNN with 3 Convolutional Layers


```python
cnn2 = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    Dropout(0.2),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])
```


```python
cnn2.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])
```


```python
cnn2.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=10,
          verbose=1,
          validation_data=(X_val, y_val))
```

    Train on 48000 samples, validate on 12000 samples
    Epoch 1/10
    48000/48000 [==============================] - 1s - loss: 0.9809 - acc: 0.6365 - val_loss: 0.5820 - val_acc: 0.7757
    Epoch 2/10
    48000/48000 [==============================] - 1s - loss: 0.5837 - acc: 0.7796 - val_loss: 0.4740 - val_acc: 0.8273
    Epoch 3/10
    48000/48000 [==============================] - 1s - loss: 0.4999 - acc: 0.8146 - val_loss: 0.4217 - val_acc: 0.8484
    Epoch 4/10
    48000/48000 [==============================] - 1s - loss: 0.4506 - acc: 0.8331 - val_loss: 0.3986 - val_acc: 0.8590
    Epoch 5/10
    48000/48000 [==============================] - 1s - loss: 0.4136 - acc: 0.8469 - val_loss: 0.3570 - val_acc: 0.8728
    Epoch 6/10
    48000/48000 [==============================] - 1s - loss: 0.3802 - acc: 0.8588 - val_loss: 0.3243 - val_acc: 0.8816
    Epoch 7/10
    48000/48000 [==============================] - 1s - loss: 0.3668 - acc: 0.8646 - val_loss: 0.3143 - val_acc: 0.8849
    Epoch 8/10
    48000/48000 [==============================] - 1s - loss: 0.3488 - acc: 0.8702 - val_loss: 0.2980 - val_acc: 0.8918
    Epoch 9/10
    48000/48000 [==============================] - 1s - loss: 0.3339 - acc: 0.8766 - val_loss: 0.2879 - val_acc: 0.8955
    Epoch 10/10
    48000/48000 [==============================] - 1s - loss: 0.3243 - acc: 0.8804 - val_loss: 0.2809 - val_acc: 0.8990









```python
cnn2.optimizer.lr = 0.0001
```


```python
cnn2.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=10,
          verbose=1,
          validation_data=(X_val, y_val))
```

    Train on 48000 samples, validate on 12000 samples
    Epoch 1/10
    48000/48000 [==============================] - 1s - loss: 0.3096 - acc: 0.8857 - val_loss: 0.2743 - val_acc: 0.9002
    Epoch 2/10
    48000/48000 [==============================] - 1s - loss: 0.2982 - acc: 0.8890 - val_loss: 0.2716 - val_acc: 0.8997
    Epoch 3/10
    48000/48000 [==============================] - 1s - loss: 0.2944 - acc: 0.8909 - val_loss: 0.2588 - val_acc: 0.9082
    Epoch 4/10
    48000/48000 [==============================] - 1s - loss: 0.2877 - acc: 0.8941 - val_loss: 0.2554 - val_acc: 0.9077
    Epoch 5/10
    48000/48000 [==============================] - 1s - loss: 0.2768 - acc: 0.8965 - val_loss: 0.2491 - val_acc: 0.9096
    Epoch 6/10
    48000/48000 [==============================] - 1s - loss: 0.2711 - acc: 0.8995 - val_loss: 0.2455 - val_acc: 0.9097
    Epoch 7/10
    48000/48000 [==============================] - 1s - loss: 0.2644 - acc: 0.9017 - val_loss: 0.2513 - val_acc: 0.9086
    Epoch 8/10
    48000/48000 [==============================] - 1s - loss: 0.2599 - acc: 0.9044 - val_loss: 0.2349 - val_acc: 0.9148
    Epoch 9/10
    48000/48000 [==============================] - 1s - loss: 0.2551 - acc: 0.9060 - val_loss: 0.2319 - val_acc: 0.9153
    Epoch 10/10
    48000/48000 [==============================] - 1s - loss: 0.2453 - acc: 0.9081 - val_loss: 0.2335 - val_acc: 0.9142









```python
score = cnn2.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

    Test loss: 0.257787227988
    Test accuracy: 0.9062


#### Data Augmentation


```python
cnn2.fit_generator(batches, steps_per_epoch=48000//batch_size, epochs=50, 
                    validation_data=val_batches, validation_steps=12000//batch_size, use_multiprocessing=True)
```

    Epoch 1/50
    93/93 [==============================] - 7s - loss: 0.4677 - acc: 0.8254 - val_loss: 0.3842 - val_acc: 0.8568
    Epoch 2/50
    93/93 [==============================] - 6s - loss: 0.4257 - acc: 0.8413 - val_loss: 0.3541 - val_acc: 0.8660
    Epoch 3/50
    93/93 [==============================] - 6s - loss: 0.4153 - acc: 0.8453 - val_loss: 0.3463 - val_acc: 0.8733
    Epoch 4/50
    93/93 [==============================] - 6s - loss: 0.3991 - acc: 0.8506 - val_loss: 0.3464 - val_acc: 0.8717
    Epoch 5/50
    93/93 [==============================] - 6s - loss: 0.3878 - acc: 0.8551 - val_loss: 0.3366 - val_acc: 0.8730
    Epoch 6/50
    93/93 [==============================] - 6s - loss: 0.3780 - acc: 0.8588 - val_loss: 0.3250 - val_acc: 0.8806
    Epoch 7/50
    93/93 [==============================] - 6s - loss: 0.3791 - acc: 0.8592 - val_loss: 0.3206 - val_acc: 0.8789
    Epoch 8/50
    93/93 [==============================] - 6s - loss: 0.3718 - acc: 0.8615 - val_loss: 0.3215 - val_acc: 0.8813
    Epoch 9/50
    93/93 [==============================] - 6s - loss: 0.3691 - acc: 0.8623 - val_loss: 0.3182 - val_acc: 0.8836
    Epoch 10/50
    93/93 [==============================] - 6s - loss: 0.3601 - acc: 0.8652 - val_loss: 0.3113 - val_acc: 0.8838
    Epoch 11/50
    93/93 [==============================] - 6s - loss: 0.3546 - acc: 0.8660 - val_loss: 0.3052 - val_acc: 0.8872
    Epoch 12/50
    93/93 [==============================] - 6s - loss: 0.3539 - acc: 0.8680 - val_loss: 0.3009 - val_acc: 0.8883
    Epoch 13/50
    93/93 [==============================] - 6s - loss: 0.3437 - acc: 0.8707 - val_loss: 0.3040 - val_acc: 0.8858
    Epoch 14/50
    93/93 [==============================] - 6s - loss: 0.3463 - acc: 0.8689 - val_loss: 0.2934 - val_acc: 0.8889
    Epoch 15/50
    93/93 [==============================] - 6s - loss: 0.3468 - acc: 0.8702 - val_loss: 0.2987 - val_acc: 0.8901
    Epoch 16/50
    93/93 [==============================] - 6s - loss: 0.3376 - acc: 0.8729 - val_loss: 0.2883 - val_acc: 0.8935
    Epoch 17/50
    93/93 [==============================] - 6s - loss: 0.3380 - acc: 0.8738 - val_loss: 0.2931 - val_acc: 0.8932
    Epoch 18/50
    93/93 [==============================] - 6s - loss: 0.3338 - acc: 0.8760 - val_loss: 0.2919 - val_acc: 0.8910
    Epoch 19/50
    93/93 [==============================] - 6s - loss: 0.3357 - acc: 0.8749 - val_loss: 0.2833 - val_acc: 0.8953
    Epoch 20/50
    93/93 [==============================] - 6s - loss: 0.3305 - acc: 0.8776 - val_loss: 0.2789 - val_acc: 0.8959
    Epoch 21/50
    93/93 [==============================] - 6s - loss: 0.3311 - acc: 0.8757 - val_loss: 0.2913 - val_acc: 0.8944
    Epoch 22/50
    93/93 [==============================] - 6s - loss: 0.3299 - acc: 0.8770 - val_loss: 0.2860 - val_acc: 0.8914
    Epoch 23/50
    93/93 [==============================] - 6s - loss: 0.3234 - acc: 0.8789 - val_loss: 0.2869 - val_acc: 0.8956
    Epoch 24/50
    93/93 [==============================] - 6s - loss: 0.3294 - acc: 0.8776 - val_loss: 0.2874 - val_acc: 0.8915
    Epoch 25/50
    93/93 [==============================] - 6s - loss: 0.3222 - acc: 0.8796 - val_loss: 0.2835 - val_acc: 0.8950
    Epoch 26/50
    93/93 [==============================] - 6s - loss: 0.3160 - acc: 0.8819 - val_loss: 0.2751 - val_acc: 0.8989
    Epoch 27/50
    93/93 [==============================] - 6s - loss: 0.3160 - acc: 0.8837 - val_loss: 0.2840 - val_acc: 0.8929
    Epoch 28/50
    93/93 [==============================] - 6s - loss: 0.3195 - acc: 0.8794 - val_loss: 0.2767 - val_acc: 0.8956
    Epoch 29/50
    93/93 [==============================] - 6s - loss: 0.3144 - acc: 0.8839 - val_loss: 0.2840 - val_acc: 0.8968
    Epoch 30/50
    93/93 [==============================] - 6s - loss: 0.3145 - acc: 0.8811 - val_loss: 0.2778 - val_acc: 0.9019
    Epoch 31/50
    93/93 [==============================] - 6s - loss: 0.3121 - acc: 0.8837 - val_loss: 0.2781 - val_acc: 0.8992
    Epoch 32/50
    93/93 [==============================] - 6s - loss: 0.3076 - acc: 0.8851 - val_loss: 0.2711 - val_acc: 0.8990
    Epoch 33/50
    93/93 [==============================] - 6s - loss: 0.3139 - acc: 0.8833 - val_loss: 0.2679 - val_acc: 0.8989
    Epoch 34/50
    93/93 [==============================] - 6s - loss: 0.3143 - acc: 0.8815 - val_loss: 0.2708 - val_acc: 0.9010
    Epoch 35/50
    93/93 [==============================] - 6s - loss: 0.3055 - acc: 0.8855 - val_loss: 0.2700 - val_acc: 0.8994
    Epoch 36/50
    93/93 [==============================] - 6s - loss: 0.3062 - acc: 0.8864 - val_loss: 0.2654 - val_acc: 0.9036
    Epoch 37/50
    93/93 [==============================] - 6s - loss: 0.3021 - acc: 0.8865 - val_loss: 0.2655 - val_acc: 0.8998
    Epoch 38/50
    93/93 [==============================] - 6s - loss: 0.3119 - acc: 0.8831 - val_loss: 0.2629 - val_acc: 0.9020
    Epoch 39/50
    93/93 [==============================] - 6s - loss: 0.3003 - acc: 0.8877 - val_loss: 0.2636 - val_acc: 0.9029
    Epoch 40/50
    93/93 [==============================] - 6s - loss: 0.3012 - acc: 0.8873 - val_loss: 0.2561 - val_acc: 0.9063
    Epoch 41/50
    93/93 [==============================] - 6s - loss: 0.2985 - acc: 0.8885 - val_loss: 0.2719 - val_acc: 0.9000
    Epoch 42/50
    93/93 [==============================] - 6s - loss: 0.3029 - acc: 0.8868 - val_loss: 0.2680 - val_acc: 0.8973
    Epoch 43/50
    93/93 [==============================] - 6s - loss: 0.2943 - acc: 0.8905 - val_loss: 0.2607 - val_acc: 0.9025
    Epoch 44/50
    93/93 [==============================] - 6s - loss: 0.3025 - acc: 0.8880 - val_loss: 0.2619 - val_acc: 0.9014
    Epoch 45/50
    93/93 [==============================] - 6s - loss: 0.2945 - acc: 0.8889 - val_loss: 0.2552 - val_acc: 0.9069
    Epoch 46/50
    93/93 [==============================] - 6s - loss: 0.2989 - acc: 0.8878 - val_loss: 0.2603 - val_acc: 0.9040
    Epoch 47/50
    93/93 [==============================] - 6s - loss: 0.2977 - acc: 0.8886 - val_loss: 0.2552 - val_acc: 0.9050
    Epoch 48/50
    93/93 [==============================] - 6s - loss: 0.2958 - acc: 0.8883 - val_loss: 0.2574 - val_acc: 0.9062
    Epoch 49/50
    93/93 [==============================] - 6s - loss: 0.2888 - acc: 0.8942 - val_loss: 0.2915 - val_acc: 0.8916
    Epoch 50/50
    93/93 [==============================] - 6s - loss: 0.2897 - acc: 0.8912 - val_loss: 0.2569 - val_acc: 0.9041









```python
score = cnn2.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

    Test loss: 0.239840124524
    Test accuracy: 0.9095


## CNN with 4 Convolutional Layers and Batch Normalization


```python
mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)
def norm_input(x): return (x-mean_px)/std_px
```


```python
cnn3 = Sequential([
    Lambda(norm_input, input_shape=(28,28, 1)),
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    BatchNormalization(),

    Conv2D(32, kernel_size=(3, 3), activation='relu'),    
    BatchNormalization(),
    Dropout(0.25),

    Conv2D(64, kernel_size=(3, 3), activation='relu'),    
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    
    Conv2D(128, kernel_size=(3, 3), activation='relu'),    
    BatchNormalization(),
    Dropout(0.25),

    Flatten(),

    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```


```python
cnn3.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])
```


```python
cnn3.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=10,
          verbose=1,
          validation_data=(X_val, y_val))
```

    Train on 48000 samples, validate on 12000 samples
    Epoch 1/10
    48000/48000 [==============================] - 5s - loss: 0.7104 - acc: 0.7530 - val_loss: 1.9009 - val_acc: 0.5357
    Epoch 2/10
    48000/48000 [==============================] - 5s - loss: 0.4277 - acc: 0.8448 - val_loss: 1.8746 - val_acc: 0.5033
    Epoch 3/10
    48000/48000 [==============================] - 5s - loss: 0.3553 - acc: 0.8730 - val_loss: 1.6118 - val_acc: 0.5543
    Epoch 4/10
    48000/48000 [==============================] - 5s - loss: 0.3102 - acc: 0.8877 - val_loss: 0.8439 - val_acc: 0.7046
    Epoch 5/10
    48000/48000 [==============================] - 5s - loss: 0.2814 - acc: 0.8984 - val_loss: 0.4175 - val_acc: 0.8534
    Epoch 6/10
    48000/48000 [==============================] - 5s - loss: 0.2582 - acc: 0.9079 - val_loss: 0.2650 - val_acc: 0.9050
    Epoch 7/10
    48000/48000 [==============================] - 5s - loss: 0.2423 - acc: 0.9142 - val_loss: 0.2335 - val_acc: 0.9178
    Epoch 8/10
    48000/48000 [==============================] - 5s - loss: 0.2272 - acc: 0.9184 - val_loss: 0.2457 - val_acc: 0.9127
    Epoch 9/10
    48000/48000 [==============================] - 5s - loss: 0.2123 - acc: 0.9234 - val_loss: 0.2254 - val_acc: 0.9214
    Epoch 10/10
    48000/48000 [==============================] - 5s - loss: 0.1998 - acc: 0.9287 - val_loss: 0.2243 - val_acc: 0.9230









```python
cnn3.optimizer.lr = 0.0001
```


```python
cnn3.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=10,
          verbose=1,
          validation_data=(X_val, y_val))
```

    Train on 48000 samples, validate on 12000 samples
    Epoch 1/10
    48000/48000 [==============================] - 5s - loss: 0.1881 - acc: 0.9314 - val_loss: 0.2101 - val_acc: 0.9270
    Epoch 2/10
    48000/48000 [==============================] - 5s - loss: 0.1753 - acc: 0.9362 - val_loss: 0.1913 - val_acc: 0.9338
    Epoch 3/10
    48000/48000 [==============================] - 5s - loss: 0.1707 - acc: 0.9380 - val_loss: 0.2064 - val_acc: 0.9291
    Epoch 4/10
    48000/48000 [==============================] - 5s - loss: 0.1570 - acc: 0.9438 - val_loss: 0.1977 - val_acc: 0.9312
    Epoch 5/10
    48000/48000 [==============================] - 5s - loss: 0.1567 - acc: 0.9428 - val_loss: 0.1824 - val_acc: 0.9376
    Epoch 6/10
    48000/48000 [==============================] - 5s - loss: 0.1420 - acc: 0.9480 - val_loss: 0.1919 - val_acc: 0.9358
    Epoch 7/10
    48000/48000 [==============================] - 5s - loss: 0.1342 - acc: 0.9506 - val_loss: 0.1856 - val_acc: 0.9373
    Epoch 8/10
    48000/48000 [==============================] - 5s - loss: 0.1313 - acc: 0.9519 - val_loss: 0.2004 - val_acc: 0.9328
    Epoch 9/10
    48000/48000 [==============================] - 5s - loss: 0.1229 - acc: 0.9562 - val_loss: 0.1984 - val_acc: 0.9350
    Epoch 10/10
    48000/48000 [==============================] - 5s - loss: 0.1192 - acc: 0.9552 - val_loss: 0.2071 - val_acc: 0.9354









```python
score = cnn3.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

    Test loss: 0.239513599713
    Test accuracy: 0.9267


#### Data Augmentation


```python
cnn3.fit_generator(batches, steps_per_epoch=48000//batch_size, epochs=50, 
                    validation_data=val_batches, validation_steps=12000//batch_size, use_multiprocessing=True)
```

    Epoch 1/50
    93/93 [==============================] - 7s - loss: 0.4569 - acc: 0.8427 - val_loss: 0.3263 - val_acc: 0.8827
    Epoch 2/50
    93/93 [==============================] - 6s - loss: 0.3597 - acc: 0.8688 - val_loss: 0.3222 - val_acc: 0.8842
    Epoch 3/50
    93/93 [==============================] - 6s - loss: 0.3402 - acc: 0.8768 - val_loss: 0.2789 - val_acc: 0.8995
    Epoch 4/50
    93/93 [==============================] - 6s - loss: 0.3233 - acc: 0.8829 - val_loss: 0.2744 - val_acc: 0.9009
    Epoch 5/50
    93/93 [==============================] - 6s - loss: 0.3090 - acc: 0.8894 - val_loss: 0.2834 - val_acc: 0.8992
    Epoch 6/50
    93/93 [==============================] - 6s - loss: 0.3091 - acc: 0.8880 - val_loss: 0.2749 - val_acc: 0.8991
    Epoch 7/50
    93/93 [==============================] - 6s - loss: 0.3000 - acc: 0.8918 - val_loss: 0.2589 - val_acc: 0.9056
    Epoch 8/50
    93/93 [==============================] - 6s - loss: 0.2883 - acc: 0.8963 - val_loss: 0.2549 - val_acc: 0.9079
    Epoch 9/50
    93/93 [==============================] - 6s - loss: 0.2906 - acc: 0.8939 - val_loss: 0.2541 - val_acc: 0.9084
    Epoch 10/50
    93/93 [==============================] - 6s - loss: 0.2881 - acc: 0.8962 - val_loss: 0.2586 - val_acc: 0.9065
    Epoch 11/50
    93/93 [==============================] - 6s - loss: 0.2837 - acc: 0.8971 - val_loss: 0.2782 - val_acc: 0.9019
    Epoch 12/50
    93/93 [==============================] - 6s - loss: 0.2740 - acc: 0.8998 - val_loss: 0.2354 - val_acc: 0.9142
    Epoch 13/50
    93/93 [==============================] - 6s - loss: 0.2773 - acc: 0.8998 - val_loss: 0.2586 - val_acc: 0.9047
    Epoch 14/50
    93/93 [==============================] - 6s - loss: 0.2705 - acc: 0.9016 - val_loss: 0.2306 - val_acc: 0.9164
    Epoch 15/50
    93/93 [==============================] - 6s - loss: 0.2662 - acc: 0.9020 - val_loss: 0.2401 - val_acc: 0.9141
    Epoch 16/50
    93/93 [==============================] - 6s - loss: 0.2643 - acc: 0.9055 - val_loss: 0.2393 - val_acc: 0.9127
    Epoch 17/50
    93/93 [==============================] - 6s - loss: 0.2613 - acc: 0.9046 - val_loss: 0.2363 - val_acc: 0.9140
    Epoch 18/50
    93/93 [==============================] - 6s - loss: 0.2594 - acc: 0.9070 - val_loss: 0.2379 - val_acc: 0.9183
    Epoch 19/50
    93/93 [==============================] - 6s - loss: 0.2566 - acc: 0.9072 - val_loss: 0.2533 - val_acc: 0.9110
    Epoch 20/50
    93/93 [==============================] - 6s - loss: 0.2528 - acc: 0.9079 - val_loss: 0.2258 - val_acc: 0.9195
    Epoch 21/50
    93/93 [==============================] - 6s - loss: 0.2514 - acc: 0.9089 - val_loss: 0.2231 - val_acc: 0.9193
    Epoch 22/50
    93/93 [==============================] - 6s - loss: 0.2485 - acc: 0.9090 - val_loss: 0.2231 - val_acc: 0.9219
    Epoch 23/50
    93/93 [==============================] - 6s - loss: 0.2475 - acc: 0.9097 - val_loss: 0.2255 - val_acc: 0.9160
    Epoch 24/50
    93/93 [==============================] - 6s - loss: 0.2506 - acc: 0.9087 - val_loss: 0.2193 - val_acc: 0.9214
    Epoch 25/50
    93/93 [==============================] - 6s - loss: 0.2423 - acc: 0.9122 - val_loss: 0.2204 - val_acc: 0.9198
    Epoch 26/50
    93/93 [==============================] - 6s - loss: 0.2466 - acc: 0.9114 - val_loss: 0.2260 - val_acc: 0.9184
    Epoch 27/50
    93/93 [==============================] - 6s - loss: 0.2466 - acc: 0.9109 - val_loss: 0.2190 - val_acc: 0.9198
    Epoch 28/50
    93/93 [==============================] - 6s - loss: 0.2416 - acc: 0.9131 - val_loss: 0.2286 - val_acc: 0.9179
    Epoch 29/50
    93/93 [==============================] - 6s - loss: 0.2394 - acc: 0.9131 - val_loss: 0.2235 - val_acc: 0.9209
    Epoch 30/50
    93/93 [==============================] - 6s - loss: 0.2308 - acc: 0.9162 - val_loss: 0.2232 - val_acc: 0.9185
    Epoch 31/50
    93/93 [==============================] - 6s - loss: 0.2370 - acc: 0.9134 - val_loss: 0.2037 - val_acc: 0.9250
    Epoch 32/50
    93/93 [==============================] - 6s - loss: 0.2301 - acc: 0.9154 - val_loss: 0.2169 - val_acc: 0.9228
    Epoch 33/50
    93/93 [==============================] - 6s - loss: 0.2315 - acc: 0.9155 - val_loss: 0.2172 - val_acc: 0.9211
    Epoch 34/50
    93/93 [==============================] - 6s - loss: 0.2280 - acc: 0.9172 - val_loss: 0.2191 - val_acc: 0.9213
    Epoch 35/50
    93/93 [==============================] - 6s - loss: 0.2302 - acc: 0.9157 - val_loss: 0.2095 - val_acc: 0.9261
    Epoch 36/50
    93/93 [==============================] - 6s - loss: 0.2227 - acc: 0.9192 - val_loss: 0.2259 - val_acc: 0.9201
    Epoch 37/50
    93/93 [==============================] - 6s - loss: 0.2275 - acc: 0.9164 - val_loss: 0.2080 - val_acc: 0.9260
    Epoch 38/50
    93/93 [==============================] - 6s - loss: 0.2257 - acc: 0.9176 - val_loss: 0.2092 - val_acc: 0.9281
    Epoch 39/50
    93/93 [==============================] - 6s - loss: 0.2234 - acc: 0.9197 - val_loss: 0.2032 - val_acc: 0.9294
    Epoch 40/50
    93/93 [==============================] - 6s - loss: 0.2223 - acc: 0.9191 - val_loss: 0.2255 - val_acc: 0.9201
    Epoch 41/50
    93/93 [==============================] - 6s - loss: 0.2234 - acc: 0.9208 - val_loss: 0.2109 - val_acc: 0.9254
    Epoch 42/50
    93/93 [==============================] - 6s - loss: 0.2185 - acc: 0.9216 - val_loss: 0.2096 - val_acc: 0.9240
    Epoch 43/50
    93/93 [==============================] - 6s - loss: 0.2120 - acc: 0.9232 - val_loss: 0.2235 - val_acc: 0.9230
    Epoch 44/50
    93/93 [==============================] - 6s - loss: 0.2155 - acc: 0.9213 - val_loss: 0.2136 - val_acc: 0.9251
    Epoch 45/50
    93/93 [==============================] - 6s - loss: 0.2165 - acc: 0.9223 - val_loss: 0.2199 - val_acc: 0.9226
    Epoch 46/50
    93/93 [==============================] - 6s - loss: 0.2168 - acc: 0.9206 - val_loss: 0.2117 - val_acc: 0.9243
    Epoch 47/50
    93/93 [==============================] - 6s - loss: 0.2140 - acc: 0.9228 - val_loss: 0.2175 - val_acc: 0.9243
    Epoch 48/50
    93/93 [==============================] - 6s - loss: 0.2112 - acc: 0.9234 - val_loss: 0.2141 - val_acc: 0.9249
    Epoch 49/50
    93/93 [==============================] - 6s - loss: 0.2082 - acc: 0.9245 - val_loss: 0.2084 - val_acc: 0.9247
    Epoch 50/50
    93/93 [==============================] - 6s - loss: 0.2127 - acc: 0.9222 - val_loss: 0.2031 - val_acc: 0.9275









```python
score = cnn3.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

    Test loss: 0.188260396481
    Test accuracy: 0.9354


## VGG Like Model


```python
cnn4 = Sequential([
    Lambda(norm_input, input_shape=(28,28, 1)),
    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    
    
    Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),    
    MaxPooling2D(pool_size=(2, 2)),
    
    
    Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
    Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),    
    MaxPooling2D(pool_size=(2, 2)),
    
    
    Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
    Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
    Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),

    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```


```python
cnn4.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])
```


```python
cnn4.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=10,
          verbose=1,
          validation_data=(X_val, y_val))
```

    Train on 48000 samples, validate on 12000 samples
    Epoch 1/10
    48000/48000 [==============================] - 4s - loss: 1.1620 - acc: 0.5509 - val_loss: 0.5807 - val_acc: 0.7721
    Epoch 2/10
    48000/48000 [==============================] - 3s - loss: 0.5223 - acc: 0.8059 - val_loss: 0.4136 - val_acc: 0.8458
    Epoch 3/10
    48000/48000 [==============================] - 3s - loss: 0.3864 - acc: 0.8613 - val_loss: 0.2957 - val_acc: 0.8952
    Epoch 4/10
    48000/48000 [==============================] - 3s - loss: 0.3172 - acc: 0.8871 - val_loss: 0.3052 - val_acc: 0.8885
    Epoch 5/10
    48000/48000 [==============================] - 3s - loss: 0.2787 - acc: 0.9008 - val_loss: 0.2342 - val_acc: 0.9158
    Epoch 6/10
    48000/48000 [==============================] - 3s - loss: 0.2430 - acc: 0.9131 - val_loss: 0.2404 - val_acc: 0.9127
    Epoch 7/10
    48000/48000 [==============================] - 3s - loss: 0.2172 - acc: 0.9227 - val_loss: 0.2469 - val_acc: 0.9091
    Epoch 8/10
    48000/48000 [==============================] - 3s - loss: 0.1968 - acc: 0.9295 - val_loss: 0.2172 - val_acc: 0.9247
    Epoch 9/10
    48000/48000 [==============================] - 3s - loss: 0.1775 - acc: 0.9363 - val_loss: 0.2124 - val_acc: 0.9259
    Epoch 10/10
    48000/48000 [==============================] - 3s - loss: 0.1616 - acc: 0.9430 - val_loss: 0.2285 - val_acc: 0.9232









```python
cnn4.optimizer.lr = 0.0001
```


```python
cnn4.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=10,
          verbose=1,
          validation_data=(X_val, y_val))
```

    Train on 48000 samples, validate on 12000 samples
    Epoch 1/10
    48000/48000 [==============================] - 3s - loss: 0.1584 - acc: 0.9442 - val_loss: 0.2155 - val_acc: 0.9267
    Epoch 2/10
    48000/48000 [==============================] - 3s - loss: 0.1319 - acc: 0.9526 - val_loss: 0.2024 - val_acc: 0.9305
    Epoch 3/10
    48000/48000 [==============================] - 3s - loss: 0.1228 - acc: 0.9567 - val_loss: 0.2117 - val_acc: 0.9301
    Epoch 4/10
    48000/48000 [==============================] - 3s - loss: 0.1096 - acc: 0.9611 - val_loss: 0.2452 - val_acc: 0.9255
    Epoch 5/10
    48000/48000 [==============================] - 3s - loss: 0.0986 - acc: 0.9651 - val_loss: 0.2530 - val_acc: 0.9255
    Epoch 6/10
    48000/48000 [==============================] - 3s - loss: 0.0896 - acc: 0.9675 - val_loss: 0.2428 - val_acc: 0.9273
    Epoch 7/10
    48000/48000 [==============================] - 3s - loss: 0.0835 - acc: 0.9711 - val_loss: 0.2585 - val_acc: 0.9183
    Epoch 8/10
    48000/48000 [==============================] - 3s - loss: 0.0807 - acc: 0.9721 - val_loss: 0.2648 - val_acc: 0.9243
    Epoch 9/10
    48000/48000 [==============================] - 3s - loss: 0.0709 - acc: 0.9757 - val_loss: 0.2641 - val_acc: 0.9246
    Epoch 10/10
    48000/48000 [==============================] - 3s - loss: 0.0609 - acc: 0.9792 - val_loss: 0.2733 - val_acc: 0.9290









```python
score = cnn4.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

    Test loss: 0.334026500632
    Test accuracy: 0.9208


#### Data Augmentation


```python
cnn4.fit_generator(batches, steps_per_epoch=48000//batch_size, epochs=50, 
                    validation_data=val_batches, validation_steps=12000//batch_size, use_multiprocessing=True)
```

    Epoch 1/50
    93/93 [==============================] - 7s - loss: 0.3967 - acc: 0.8585 - val_loss: 0.3305 - val_acc: 0.8817
    Epoch 2/50
    93/93 [==============================] - 6s - loss: 0.3177 - acc: 0.8850 - val_loss: 0.2868 - val_acc: 0.8943
    Epoch 3/50
    93/93 [==============================] - 6s - loss: 0.2959 - acc: 0.8916 - val_loss: 0.2782 - val_acc: 0.8997
    Epoch 4/50
    93/93 [==============================] - 6s - loss: 0.2750 - acc: 0.8993 - val_loss: 0.2831 - val_acc: 0.8998
    Epoch 5/50
    93/93 [==============================] - 6s - loss: 0.2683 - acc: 0.9019 - val_loss: 0.2666 - val_acc: 0.9006
    Epoch 6/50
    93/93 [==============================] - 6s - loss: 0.2647 - acc: 0.9048 - val_loss: 0.2718 - val_acc: 0.9016
    Epoch 7/50
    93/93 [==============================] - 6s - loss: 0.2559 - acc: 0.9077 - val_loss: 0.2533 - val_acc: 0.9083
    Epoch 8/50
    93/93 [==============================] - 6s - loss: 0.2460 - acc: 0.9118 - val_loss: 0.2505 - val_acc: 0.9098
    Epoch 9/50
    93/93 [==============================] - 6s - loss: 0.2431 - acc: 0.9110 - val_loss: 0.2588 - val_acc: 0.9079
    Epoch 10/50
    93/93 [==============================] - 6s - loss: 0.2345 - acc: 0.9147 - val_loss: 0.2492 - val_acc: 0.9082
    Epoch 11/50
    93/93 [==============================] - 6s - loss: 0.2308 - acc: 0.9148 - val_loss: 0.2551 - val_acc: 0.9057
    Epoch 12/50
    93/93 [==============================] - 6s - loss: 0.2293 - acc: 0.9151 - val_loss: 0.2639 - val_acc: 0.9021
    Epoch 13/50
    93/93 [==============================] - 6s - loss: 0.2217 - acc: 0.9187 - val_loss: 0.2338 - val_acc: 0.9147
    Epoch 14/50
    93/93 [==============================] - 6s - loss: 0.2196 - acc: 0.9192 - val_loss: 0.2344 - val_acc: 0.9149
    Epoch 15/50
    93/93 [==============================] - 6s - loss: 0.2181 - acc: 0.9213 - val_loss: 0.2492 - val_acc: 0.9109
    Epoch 16/50
    93/93 [==============================] - 6s - loss: 0.2144 - acc: 0.9224 - val_loss: 0.2393 - val_acc: 0.9164
    Epoch 17/50
    93/93 [==============================] - 6s - loss: 0.2114 - acc: 0.9228 - val_loss: 0.2315 - val_acc: 0.9151
    Epoch 18/50
    93/93 [==============================] - 6s - loss: 0.2052 - acc: 0.9261 - val_loss: 0.2350 - val_acc: 0.9182
    Epoch 19/50
    93/93 [==============================] - 6s - loss: 0.2023 - acc: 0.9247 - val_loss: 0.2437 - val_acc: 0.9113
    Epoch 20/50
    93/93 [==============================] - 6s - loss: 0.2019 - acc: 0.9258 - val_loss: 0.2239 - val_acc: 0.9179
    Epoch 21/50
    93/93 [==============================] - 6s - loss: 0.1971 - acc: 0.9275 - val_loss: 0.2308 - val_acc: 0.9188
    Epoch 22/50
    93/93 [==============================] - 6s - loss: 0.1920 - acc: 0.9306 - val_loss: 0.2253 - val_acc: 0.9204
    Epoch 23/50
    93/93 [==============================] - 6s - loss: 0.1959 - acc: 0.9270 - val_loss: 0.2286 - val_acc: 0.9192
    Epoch 24/50
    93/93 [==============================] - 6s - loss: 0.1920 - acc: 0.9305 - val_loss: 0.2180 - val_acc: 0.9247
    Epoch 25/50
    93/93 [==============================] - 6s - loss: 0.1841 - acc: 0.9336 - val_loss: 0.2269 - val_acc: 0.9213
    Epoch 26/50
    93/93 [==============================] - 6s - loss: 0.1821 - acc: 0.9338 - val_loss: 0.2240 - val_acc: 0.9203
    Epoch 27/50
    93/93 [==============================] - 6s - loss: 0.1828 - acc: 0.9332 - val_loss: 0.2176 - val_acc: 0.9240
    Epoch 28/50
    93/93 [==============================] - 6s - loss: 0.1797 - acc: 0.9347 - val_loss: 0.2208 - val_acc: 0.9231
    Epoch 29/50
    93/93 [==============================] - 6s - loss: 0.1759 - acc: 0.9353 - val_loss: 0.2167 - val_acc: 0.9243
    Epoch 30/50
    93/93 [==============================] - 6s - loss: 0.1742 - acc: 0.9359 - val_loss: 0.2223 - val_acc: 0.9212
    Epoch 31/50
    93/93 [==============================] - 6s - loss: 0.1770 - acc: 0.9359 - val_loss: 0.2138 - val_acc: 0.9252
    Epoch 32/50
    93/93 [==============================] - 6s - loss: 0.1741 - acc: 0.9366 - val_loss: 0.2222 - val_acc: 0.9243
    Epoch 33/50
    93/93 [==============================] - 6s - loss: 0.1761 - acc: 0.9362 - val_loss: 0.2237 - val_acc: 0.9243
    Epoch 34/50
    93/93 [==============================] - 6s - loss: 0.1721 - acc: 0.9367 - val_loss: 0.2379 - val_acc: 0.9192
    Epoch 35/50
    93/93 [==============================] - 6s - loss: 0.1700 - acc: 0.9388 - val_loss: 0.2586 - val_acc: 0.9165
    Epoch 36/50
    93/93 [==============================] - 6s - loss: 0.1688 - acc: 0.9379 - val_loss: 0.2301 - val_acc: 0.9246
    Epoch 37/50
    93/93 [==============================] - 6s - loss: 0.1623 - acc: 0.9408 - val_loss: 0.2289 - val_acc: 0.9229
    Epoch 38/50
    93/93 [==============================] - 6s - loss: 0.1599 - acc: 0.9416 - val_loss: 0.2339 - val_acc: 0.9186
    Epoch 39/50
    93/93 [==============================] - 6s - loss: 0.1685 - acc: 0.9394 - val_loss: 0.2205 - val_acc: 0.9230
    Epoch 40/50
    93/93 [==============================] - 6s - loss: 0.1547 - acc: 0.9433 - val_loss: 0.2132 - val_acc: 0.9253
    Epoch 41/50
    93/93 [==============================] - 6s - loss: 0.1576 - acc: 0.9417 - val_loss: 0.2412 - val_acc: 0.9207
    Epoch 42/50
    93/93 [==============================] - 6s - loss: 0.1611 - acc: 0.9409 - val_loss: 0.2170 - val_acc: 0.9265
    Epoch 43/50
    93/93 [==============================] - 6s - loss: 0.1576 - acc: 0.9423 - val_loss: 0.2217 - val_acc: 0.9259
    Epoch 44/50
    93/93 [==============================] - 7s - loss: 0.1589 - acc: 0.9422 - val_loss: 0.2390 - val_acc: 0.9197
    Epoch 45/50
    93/93 [==============================] - 6s - loss: 0.1543 - acc: 0.9445 - val_loss: 0.2337 - val_acc: 0.9207
    Epoch 46/50
    93/93 [==============================] - 6s - loss: 0.1480 - acc: 0.9457 - val_loss: 0.2172 - val_acc: 0.9267
    Epoch 47/50
    93/93 [==============================] - 6s - loss: 0.1498 - acc: 0.9436 - val_loss: 0.2226 - val_acc: 0.9230
    Epoch 48/50
    93/93 [==============================] - 6s - loss: 0.1492 - acc: 0.9463 - val_loss: 0.2413 - val_acc: 0.9227
    Epoch 49/50
    93/93 [==============================] - 6s - loss: 0.1502 - acc: 0.9462 - val_loss: 0.2225 - val_acc: 0.9283
    Epoch 50/50
    93/93 [==============================] - 6s - loss: 0.1499 - acc: 0.9453 - val_loss: 0.2208 - val_acc: 0.9260









```python
score = cnn4.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

    Test loss: 0.193522232082
    Test accuracy: 0.9359




## VGG Like Model With Batchnorm


```python
cnn5 = Sequential([
    Lambda(norm_input, input_shape=(28,28, 1)),
    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.25),
    
    Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),    
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
    Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),    
    BatchNormalization(),
    Dropout(0.25),
    
    Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
    Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
    Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),

    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```


```python
cnn5.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])
```


```python
cnn5.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=10,
          verbose=1,
          validation_data=(X_val, y_val))
```

    Train on 48000 samples, validate on 12000 samples
    Epoch 1/10
    48000/48000 [==============================] - 17s - loss: 0.8053 - acc: 0.7310 - val_loss: 2.8340 - val_acc: 0.1061
    Epoch 2/10
    48000/48000 [==============================] - 16s - loss: 0.4330 - acc: 0.8426 - val_loss: 3.4900 - val_acc: 0.3552
    Epoch 3/10
    48000/48000 [==============================] - 16s - loss: 0.3449 - acc: 0.8752 - val_loss: 4.8988 - val_acc: 0.1126
    Epoch 4/10
    48000/48000 [==============================] - 16s - loss: 0.2972 - acc: 0.8927 - val_loss: 2.3724 - val_acc: 0.4153
    Epoch 5/10
    48000/48000 [==============================] - 16s - loss: 0.2636 - acc: 0.9033 - val_loss: 0.3967 - val_acc: 0.8660
    Epoch 6/10
    48000/48000 [==============================] - 16s - loss: 0.2409 - acc: 0.9116 - val_loss: 0.4278 - val_acc: 0.8573
    Epoch 7/10
    48000/48000 [==============================] - 16s - loss: 0.2278 - acc: 0.9169 - val_loss: 0.2007 - val_acc: 0.9286
    Epoch 8/10
    48000/48000 [==============================] - 16s - loss: 0.2058 - acc: 0.9254 - val_loss: 0.1954 - val_acc: 0.9310
    Epoch 9/10
    48000/48000 [==============================] - 16s - loss: 0.1975 - acc: 0.9268 - val_loss: 0.2283 - val_acc: 0.9186
    Epoch 10/10
    48000/48000 [==============================] - 16s - loss: 0.1852 - acc: 0.9319 - val_loss: 0.1994 - val_acc: 0.9333









```python
cnn5.optimizer.lr = 0.0001
```


```python
cnn5.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=10,
          verbose=1,
          validation_data=(X_val, y_val))
```

    Train on 48000 samples, validate on 12000 samples
    Epoch 1/10
    48000/48000 [==============================] - 16s - loss: 0.1748 - acc: 0.9360 - val_loss: 0.1984 - val_acc: 0.9331
    Epoch 2/10
    48000/48000 [==============================] - 16s - loss: 0.1652 - acc: 0.9395 - val_loss: 0.1965 - val_acc: 0.9376
    Epoch 3/10
    48000/48000 [==============================] - 16s - loss: 0.1572 - acc: 0.9425 - val_loss: 0.1781 - val_acc: 0.9401
    Epoch 4/10
    48000/48000 [==============================] - 16s - loss: 0.1454 - acc: 0.9464 - val_loss: 0.1759 - val_acc: 0.9392
    Epoch 5/10
    48000/48000 [==============================] - 16s - loss: 0.1350 - acc: 0.9499 - val_loss: 0.2181 - val_acc: 0.9338
    Epoch 6/10
    48000/48000 [==============================] - 16s - loss: 0.1277 - acc: 0.9527 - val_loss: 0.1997 - val_acc: 0.9358
    Epoch 7/10
    48000/48000 [==============================] - 16s - loss: 0.1226 - acc: 0.9551 - val_loss: 0.1930 - val_acc: 0.9394
    Epoch 8/10
    48000/48000 [==============================] - 16s - loss: 0.1119 - acc: 0.9577 - val_loss: 0.2257 - val_acc: 0.9315
    Epoch 9/10
    48000/48000 [==============================] - 16s - loss: 0.1055 - acc: 0.9606 - val_loss: 0.2066 - val_acc: 0.9375
    Epoch 10/10
    48000/48000 [==============================] - 16s - loss: 0.0970 - acc: 0.9645 - val_loss: 0.1986 - val_acc: 0.9374









```python
score = cnn5.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

    Test loss: 0.227945802512
    Test accuracy: 0.9296


#### Data Augmentation


```python
cnn5.fit_generator(batches, steps_per_epoch=48000//batch_size, epochs=50, 
                    validation_data=val_batches, validation_steps=12000//batch_size, use_multiprocessing=True)
```

    Epoch 1/50
    93/93 [==============================] - 16s - loss: 0.3573 - acc: 0.8742 - val_loss: 0.3032 - val_acc: 0.8989
    Epoch 2/50
    93/93 [==============================] - 16s - loss: 0.2950 - acc: 0.8938 - val_loss: 0.2749 - val_acc: 0.9000
    Epoch 3/50
    93/93 [==============================] - 16s - loss: 0.2756 - acc: 0.8995 - val_loss: 0.3022 - val_acc: 0.8914
    Epoch 4/50
    93/93 [==============================] - 16s - loss: 0.2672 - acc: 0.9036 - val_loss: 0.2442 - val_acc: 0.9126
    Epoch 5/50
    93/93 [==============================] - 16s - loss: 0.2559 - acc: 0.9067 - val_loss: 0.2618 - val_acc: 0.9051
    Epoch 6/50
    93/93 [==============================] - 16s - loss: 0.2493 - acc: 0.9094 - val_loss: 0.2671 - val_acc: 0.9079
    Epoch 7/50
    93/93 [==============================] - 16s - loss: 0.2421 - acc: 0.9113 - val_loss: 0.2532 - val_acc: 0.9067
    Epoch 8/50
    93/93 [==============================] - 16s - loss: 0.2375 - acc: 0.9128 - val_loss: 0.2315 - val_acc: 0.9194
    Epoch 9/50
    93/93 [==============================] - 16s - loss: 0.2337 - acc: 0.9141 - val_loss: 0.2346 - val_acc: 0.9173
    Epoch 10/50
    93/93 [==============================] - 16s - loss: 0.2290 - acc: 0.9152 - val_loss: 0.2551 - val_acc: 0.9107
    Epoch 11/50
    93/93 [==============================] - 16s - loss: 0.2186 - acc: 0.9194 - val_loss: 0.2457 - val_acc: 0.9092
    Epoch 12/50
    93/93 [==============================] - 16s - loss: 0.2200 - acc: 0.9189 - val_loss: 0.2460 - val_acc: 0.9136
    Epoch 13/50
    93/93 [==============================] - 16s - loss: 0.2124 - acc: 0.9217 - val_loss: 0.2248 - val_acc: 0.9219
    Epoch 14/50
    93/93 [==============================] - 16s - loss: 0.2141 - acc: 0.9219 - val_loss: 0.2121 - val_acc: 0.9239
    Epoch 15/50
    93/93 [==============================] - 16s - loss: 0.2051 - acc: 0.9241 - val_loss: 0.2335 - val_acc: 0.9191
    Epoch 16/50
    93/93 [==============================] - 16s - loss: 0.2059 - acc: 0.9237 - val_loss: 0.2222 - val_acc: 0.9214
    Epoch 17/50
    93/93 [==============================] - 16s - loss: 0.2066 - acc: 0.9246 - val_loss: 0.2112 - val_acc: 0.9231
    Epoch 18/50
    93/93 [==============================] - 16s - loss: 0.1977 - acc: 0.9272 - val_loss: 0.2345 - val_acc: 0.9187
    Epoch 19/50
    93/93 [==============================] - 16s - loss: 0.1955 - acc: 0.9270 - val_loss: 0.1980 - val_acc: 0.9296
    Epoch 20/50
    93/93 [==============================] - 16s - loss: 0.1965 - acc: 0.9284 - val_loss: 0.2066 - val_acc: 0.9281
    Epoch 21/50
    93/93 [==============================] - 16s - loss: 0.1901 - acc: 0.9307 - val_loss: 0.2169 - val_acc: 0.9251
    Epoch 22/50
    93/93 [==============================] - 16s - loss: 0.1913 - acc: 0.9296 - val_loss: 0.2052 - val_acc: 0.9277
    Epoch 23/50
    93/93 [==============================] - 16s - loss: 0.1824 - acc: 0.9333 - val_loss: 0.2103 - val_acc: 0.9266
    Epoch 24/50
    93/93 [==============================] - 16s - loss: 0.1897 - acc: 0.9308 - val_loss: 0.2338 - val_acc: 0.9162
    Epoch 25/50
    93/93 [==============================] - 16s - loss: 0.1798 - acc: 0.9346 - val_loss: 0.2226 - val_acc: 0.9222
    Epoch 26/50
    93/93 [==============================] - 16s - loss: 0.1856 - acc: 0.9323 - val_loss: 0.2038 - val_acc: 0.9277
    Epoch 27/50
    93/93 [==============================] - 16s - loss: 0.1815 - acc: 0.9346 - val_loss: 0.2009 - val_acc: 0.9314
    Epoch 28/50
    93/93 [==============================] - 16s - loss: 0.1704 - acc: 0.9374 - val_loss: 0.2132 - val_acc: 0.9288
    Epoch 29/50
    93/93 [==============================] - 16s - loss: 0.1742 - acc: 0.9362 - val_loss: 0.2063 - val_acc: 0.9265
    Epoch 30/50
    93/93 [==============================] - 16s - loss: 0.1670 - acc: 0.9390 - val_loss: 0.2060 - val_acc: 0.9282
    Epoch 31/50
    93/93 [==============================] - 16s - loss: 0.1638 - acc: 0.9386 - val_loss: 0.1994 - val_acc: 0.9351
    Epoch 32/50
    93/93 [==============================] - 16s - loss: 0.1666 - acc: 0.9396 - val_loss: 0.2001 - val_acc: 0.9327
    Epoch 33/50
    93/93 [==============================] - 16s - loss: 0.1610 - acc: 0.9406 - val_loss: 0.2188 - val_acc: 0.9240
    Epoch 34/50
    93/93 [==============================] - 16s - loss: 0.1623 - acc: 0.9409 - val_loss: 0.1985 - val_acc: 0.9313
    Epoch 35/50
    93/93 [==============================] - 16s - loss: 0.1562 - acc: 0.9429 - val_loss: 0.2299 - val_acc: 0.9256
    Epoch 36/50
    93/93 [==============================] - 16s - loss: 0.1640 - acc: 0.9402 - val_loss: 0.2170 - val_acc: 0.9226
    Epoch 37/50
    93/93 [==============================] - 16s - loss: 0.1560 - acc: 0.9429 - val_loss: 0.1942 - val_acc: 0.9337
    Epoch 38/50
    93/93 [==============================] - 16s - loss: 0.1601 - acc: 0.9414 - val_loss: 0.2054 - val_acc: 0.9302
    Epoch 39/50
    93/93 [==============================] - 16s - loss: 0.1503 - acc: 0.9444 - val_loss: 0.1953 - val_acc: 0.9323
    Epoch 40/50
    93/93 [==============================] - 16s - loss: 0.1502 - acc: 0.9443 - val_loss: 0.2039 - val_acc: 0.9342
    Epoch 41/50
    93/93 [==============================] - 16s - loss: 0.1455 - acc: 0.9459 - val_loss: 0.2034 - val_acc: 0.9319
    Epoch 42/50
    93/93 [==============================] - 16s - loss: 0.1521 - acc: 0.9439 - val_loss: 0.1959 - val_acc: 0.9346
    Epoch 43/50
    93/93 [==============================] - 16s - loss: 0.1422 - acc: 0.9482 - val_loss: 0.2070 - val_acc: 0.9322
    Epoch 44/50
    93/93 [==============================] - 16s - loss: 0.1407 - acc: 0.9479 - val_loss: 0.2077 - val_acc: 0.9314
    Epoch 45/50
    93/93 [==============================] - 16s - loss: 0.1381 - acc: 0.9498 - val_loss: 0.2056 - val_acc: 0.9327
    Epoch 46/50
    93/93 [==============================] - 16s - loss: 0.1377 - acc: 0.9490 - val_loss: 0.2174 - val_acc: 0.9277
    Epoch 47/50
    93/93 [==============================] - 16s - loss: 0.1374 - acc: 0.9490 - val_loss: 0.2012 - val_acc: 0.9341
    Epoch 48/50
    93/93 [==============================] - 16s - loss: 0.1331 - acc: 0.9508 - val_loss: 0.2001 - val_acc: 0.9338
    Epoch 49/50
    93/93 [==============================] - 16s - loss: 0.1390 - acc: 0.9493 - val_loss: 0.1798 - val_acc: 0.9403
    Epoch 50/50
    93/93 [==============================] - 16s - loss: 0.1328 - acc: 0.9514 - val_loss: 0.1904 - val_acc: 0.9366









```python
score = cnn5.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

    Test loss: 0.177969511382
    Test accuracy: 0.9401




You can find the original notebook on my [Github](https://github.com/DanialK/fashion-mnist-cnn/blob/master/fashion-mnist-final.ipynb)


