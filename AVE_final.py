import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from scipy.stats import norm
import os
from keras.utils import plot_model

# network parameters
batch_size = 128
n_epoch = 200
z_dim = 2

# data load
(x_tr, y_tr), (x_te, y_te) = mnist.load_data()
x_tr, x_te = x_tr.astype('float32')/255., x_te.astype('float32')/255.
x_tr, x_te = x_tr.reshape(x_tr.shape[0], -1), x_te.reshape(x_te.shape[0], -1)

number = 9

#folder name
folder_name = 'image' + str(number)
#folder path
path = './' + folder_name +'/'

label_data = []
for i in range(x_tr.shape[0]):
    # check the specify label
    if y_tr[i] == number:
       label_data.append(x_tr[i])
x_tr = np.array(label_data)  

label_data2 = []
for i in range(x_te.shape[0]):
    # check the specify label
    if y_te[i] == number:
       label_data2.append(x_te[i])
x_te = np.array(label_data2)  

# create new folder
def new_folder(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

print(x_tr.shape, x_te.shape)

# encoder
x = Input(shape=(x_tr.shape[1:]))
encoder = Dense(1024, activation='relu')(x)
encoder = Dense(512, activation='relu')(encoder)
encoder = Dense(256, activation='relu')(encoder)
encoder = Dense(128, activation='relu')(encoder)
en_mean = Dense(z_dim)(encoder)
log_var = Dense(z_dim)(encoder)

def sampling(args):
    en_mean, log_var = args
    eps = K.random_normal(shape=(batch_size, z_dim), mean=0., stddev=1.0)
    return en_mean + K.exp(log_var) * eps

z = Lambda(sampling, output_shape=(z_dim,))([en_mean, log_var])


decoder = Dense(x_tr.shape[1], activation='sigmoid')
z_decoded = Dense(128, activation='relu')(z)
z_decoded = Dense(256, activation='relu')(z_decoded)
z_decoded = Dense(512, activation='relu')(z_decoded)
z_decoded = Dense(1024, activation='relu')(z_decoded)
y = decoder(z_decoded)

# loss
reconstruction_loss = objectives.binary_crossentropy(x, y) * x_tr.shape[1]
kl_loss = 0.5 * K.sum(K.square(en_mean) + K.exp(log_var) - log_var - 1, axis = -1)
vae_loss = reconstruction_loss + kl_loss

# build model
VAE = Model(x, y)
VAE.add_loss(vae_loss)
VAE.compile(optimizer='rmsprop')
VAE.summary()
plot_model(VAE, to_file='VAE_plot.png', show_shapes=True, show_layer_names=False)

size = (int(x_tr.shape[0]/batch_size)) * batch_size
VAE.fit(x_tr[:size],
       shuffle=True,
       epochs=n_epoch,
       batch_size=batch_size,
       verbose=1)

# build encoder
encoder = Model(x, en_mean)
encoder.summary()

# build decoder
decoder_input = Input(shape=(z_dim,))
_z_decoded = Dense(128, activation='relu')(decoder_input)
_z_decoded = Dense(256, activation='relu')(_z_decoded)
_z_decoded = Dense(512, activation='relu')(_z_decoded)
_z_decoded = Dense(1024, activation='relu')(_z_decoded)
_y = decoder(_z_decoded)
generator = Model(decoder_input, _y)
generator.summary()


n = 1 
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

grid_x = norm.ppf(np.linspace(0.05, 0.95, n)) 
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(1, 1))
plt.imshow(figure, cmap='Greys')
plt.axis('off')
new_folder(path)
plt.savefig(path +'images %d.png' %number)