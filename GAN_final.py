# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:46:36 2020

@author: monol
"""
import os
from keras.datasets import mnist
from keras.layers import Input
from keras.models import Model,Sequential
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
from numpy import expand_dims
import matplotlib.pyplot as plt
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Reshape
from keras.layers import Conv2DTranspose
from keras.layers import Convolution2D
from keras.layers import UpSampling2D
from keras.utils import plot_model

#epoch and brantch size
epoch = 200
batch_size = 128 
#label
number = 8

#for pic print
examples = 1
dim = (1,1)
figure_size = (10,10)
#folder name
folder_name = 'image' + str(number)
#folder path
path = './' + folder_name +'/'
#input shap
inputShape = (28,28,1)
#generator input dimention
gen_input_dim = 100


# read date
def data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32)
    x_train = (x_train - 127.5) / 127.5

    #pix = 28 * 28
    #x_train = x_train.reshape(x_train.shape[0], pix)
    x_train = expand_dims(x_train, -1)
    return (x_train, y_train)

# get a specify label
def label_data(x_train, y_train, number):
    label_data = []
    for i in range(x_train.shape[0]):
        # check the specify label
        if y_train[i] == number:
           label_data.append(x_train[i])
    label_data = np.array(label_data)  
    return label_data

# create new folder
def new_folder(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        
def build_discriminator(inputShape):
    model = Sequential()
    
    model.add(Convolution2D(64, (5, 5), strides = 2, padding = 'same', input_shape = inputShape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.2))
    
    model.add(Convolution2D(64*2, (5, 5), strides = 2, padding = 'same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.2))
    
    model.add(Convolution2D(64*4, (5, 5), strides = 2, padding = 'same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.2))
    
    model.add(Convolution2D(64*8, (5, 5), strides = 1, padding = 'same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(1, activation = 'sigmoid'))
    #model.add(activation = 'sigmoid')
    #discriminator.trainable = True
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer="SGD")
    return model

def build_generator(gen_input_dim):
    model = Sequential()

    model.add(Dense(7*7*256, input_dim = gen_input_dim, activation = 'relu'))
    model.add(BatchNormalization(momentum=0.9))
    #model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, 256)))
    
    model.add(UpSampling2D())
    model.add(Conv2DTranspose(128, (5,5), padding = 'same', activation = 'relu'))
    model.add(BatchNormalization(momentum=0.9))
    #model.add(LeakyReLU(alpha=0.2))
    
    model.add(UpSampling2D())
    model.add(Conv2DTranspose(64, (5,5), padding = 'same', activation = 'relu'))
    model.add(BatchNormalization(momentum=0.9))
    #model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2DTranspose(32, (5,5), padding = 'same', activation = 'relu'))
    model.add(BatchNormalization(momentum=0.9))
    #model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2DTranspose(1, (5,5), padding = 'same', activation = 'tanh'))
    #assert model.output_shape == (None, 28, 28, 1)
    #model.add(activation='sigmoid')

    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer="SGD")
    return model

def GAN_model(generator, discriminator):
    model = Sequential()
    
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer="SGD")
    return model

def real_image(train_label, batch_size):
     real_images = train_label[np.random.randint(0, train_label.shape[0], batch_size)]
     real_label = np.ones(batch_size) 
     return real_images, real_label  

def fake_image(batch_size, gen_input_dim, generator):
    noise= np.random.normal(0,1, (batch_size, gen_input_dim))
    #noise = noise.reshape(batch_size, 28, 28 ,1)
    fake_images = generator.predict(noise)
    fake_label = np.zeros(batch_size)
    return noise, fake_images, fake_label

def real_accuracy(train_label, batch_size, discriminator):
    X, Y = real_image(train_label, batch_size)
    _, accuracy = discriminator.evaluate(X, Y, verbose = 0)
    return accuracy

def fake_accuracy(batch_size, gen_input_dim, generator):
    noise, image, label = fake_image(batch_size, gen_input_dim, generator)
    _, acc = discriminator.evaluate(image, label, verbose = 0)
    return acc

#train GAN
#get data
x_train, y_train = data()
#get a specify label of the data
train_label = label_data(x_train, y_train, number)


discriminator = build_discriminator(inputShape)
print('generator model:')
discriminator.summary()
plot_model(discriminator, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=False)
generator = build_generator(gen_input_dim)
print('discriminator model:')
generator.summary()
plot_model(generator, to_file='generator_plot.png', show_shapes=True, show_layer_names=False)

#build GAN
discriminator.trainable = False

GAN = GAN_model(generator, discriminator)
print('GAN model:')
GAN.summary()
plot_model(GAN, to_file='GAN_plot.png', show_shapes=True, show_layer_names=False)

sum_acc = 0
for i in range (1, epoch + 1):
    each = int(train_label.shape[0] / (2 * batch_size))
    for j in range (0, each):
        #real images
        real_images, real_label = real_image(train_label, batch_size)
        
        #fake images
        noise, fake_images, fake_label = fake_image(batch_size, gen_input_dim, generator)
    
        # Concatenate fake and real images 
        cob_image = np.concatenate([fake_images, real_images])
        cob_label = np.concatenate([fake_label, real_label])
    
        # Train the discriminator
        discriminator.trainable = True
        d_loss = discriminator.train_on_batch(cob_image, cob_label)
    
        # Train the generator/chained GAN model (with frozen weights in discriminator) 
        discriminator.trainable = False
        g_loss = GAN.train_on_batch(noise, real_label)
        #print(g_loss)
   
    if i == 1 or i % 10 == 0:
          noise= np.random.normal(loc=0, scale=1, size=[examples, gen_input_dim])
          generated_images = generator.predict(noise)
          generated_images = generated_images.reshape(1,28,28)
          plt.figure(figsize = figure_size)
          for k in range(generated_images.shape[0]):
              plt.subplot(dim[0], dim[1], k+1)
              plt.imshow(generated_images[k], interpolation='nearest', cmap='Greys')
              plt.axis('off')
          plt.tight_layout()
          
          new_folder(path)
          plt.savefig(path +'images %d.png' %i)
          print('one image is printed')
   
          
          
          
          
          
          
          
          