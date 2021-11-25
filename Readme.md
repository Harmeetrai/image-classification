# Image Classification App
In this project, we will build a convolution neural network in Keras with python on a CIFAR-10 dataset. Then, we will train and build our classification model and delopy it using streamlit and heroku. Please check out the full app here: https://image-classify-app.herokuapp.com/.

## Load Dataset
We will be using the CIFAR-10 dataset that is already included in the keras datasets library:

```py
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```
Check if images are loaded:
```py
plt.figure(figsize=(20, 10))
for i in range(9):
    plt.subplot(330+1+i)
    plt.imshow(x_train[i])
    plt.title(y_train[i])
    plt.axis('off')
```
## Data Processing
### Load Libraries
```py
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
```
### Train and Test sets
Convert the pixel values of the dataset to float type and then normalize the dataset
```py
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train / 255.0
x_test = x_test / 255.0
```
One-hot encoding for target classes
```py
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]
```

## Modeling
Create the sequential model and add the layers
```py
model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(32,32,3),
    padding='same',activation='relu',
    kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32,(3,3),activation='relu',padding='same',kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512,activation='relu',kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
```
### Optimizer and Compile Model
```py
sgd = SGD(lr=0.01, momentum=0.9, decay=(0.01/25), nesterov=False)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
```
### Train Model
```py
model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=25, batch_size=32)
```
### Check Accuracy
```py
acc = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', acc[1])
```
```py
model.save('cifar10_model.h5')
```
## Testing Model
Using `test.jpg` (airplane image), we will test to see if the model works
```py
results = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

from PIL import Image
import numpy as np
img = Image.open('test.jpg')
img = img.resize((32, 32))
img = np.array(img)
img = img.astype('float32')
img = img / 255.0
img = np.expand_dims(img, axis=0)
pred = model.predict(img)
print(pred, "\n",  results[np.argmax(pred)])
```
Output:
```
[[9.9999976e-01 6.2572963e-10 5.3366811e-08 1.9468609e-10 2.0309849e-07
  3.6830453e-10 5.5412207e-12 2.0695565e-12 8.1430986e-09 1.1587767e-12]] 
 airplane
```
The Model works!

# Building Streamlit App
Using the model we saved called `cifar10_model.h5`, we will build an interactive app using streamlit. Learn more about streamlit [here](https://docs.streamlit.io/).

### app.py
This will be the main file used to run the app. Please take a look in the [github repo](https://github.com/Harmeetrai/image-classification/blob/main/app.py).

### setup.sh
This is the setup file for heroku:
```sh
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```
### Procfile
```
web: sh setup.sh && streamlit run app.py
```
### Requirements.txt
You will need a separate requirements.txt file that is dedicated to streamlit:
```txt
streamlit
Pillow==8.3.2
tensorflow-cpu
numpy
pytest-shutil
```
### Deploy to Heroku
Now deploy the code to [Heroku](https://heroku.com/). This can be done by first deploying to github and linking your github repo to Heroku. 
# Credit
Special thanks to [this](https://data-flair.training/blogs/image-classification-deep-learning-project-python-keras/) blog post for the inspiration: 