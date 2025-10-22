import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU
from keras.optimizers import Adam
(X_train,_),(_,_) = fashion_mnist.load_data()
X_train = (X_train.astype(np.float32)-127.5)/127.5
X_train = np.expand_dims(X_train, axis=-1)
latent_dim = 100
batch_size = 64
sample_interval = 2000
G = Sequential([
    Dense(128*7*7, input_shape=(latent_dim,), activation='relu'),
    Reshape((7,7,128)),
    Conv2DTranspose(128,4,strides=2,padding='same',activation='relu'),
    Conv2DTranspose(64,4,strides=2,padding='same',activation='relu'),
    Conv2DTranspose(1,7,activation='tanh',padding='same')
])
D = Sequential([
    Conv2D(64,3,strides=2,padding='same',input_shape=(28,28,1)),
    LeakyReLU(0.2),
    Conv2D(128,3,strides=2,padding='same'),
    LeakyReLU(0.2),
    Flatten(),
    Dense(1,activation='sigmoid')
])
D.compile(loss='binary_crossentropy',optimizer=Adam(0.0002,0.5),metrics=['accuracy'])
D.trainable = False
from keras.models import Model
from keras.layers import Input
z = Input(shape=(latent_dim,))
GAN = Model(z,D(G(z)))
GAN.compile(loss='binary_crossentropy',optimizer=Adam(0.0002,0.5))
valid = np.ones((batch_size,1))
fake = np.zeros((batch_size,1))
for e in range(10000):
    idx = np.random.randint(0,X_train.shape[0],batch_size)
    imgs = X_train[idx]
    noise = np.random.normal(0,1,(batch_size,latent_dim))
    gen_imgs = G.predict(noise,verbose=0)
    d_loss = 0.5*(D.train_on_batch(imgs,valid)[0]+D.train_on_batch(gen_imgs,fake)[0])
    noise = np.random.normal(0,1,(batch_size,latent_dim))
    g_loss = GAN.train_on_batch(noise,valid)
    if e % sample_interval == 0:
        print(f"{e} [D:{d_loss:.4f}] [G:{g_loss:.4f}]")
        s = np.random.normal(0,1,(9,latent_dim))
        g = G.predict(s,verbose=0); g = 0.5*g+0.5
        plt.figure(figsize=(3,3))
        for i in range(9):
            plt.subplot(3,3,i+1)
            plt.imshow(g[i,:,:,0],cmap='gray')
            plt.axis('off')
        plt.show()
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU
from keras.optimizers import Adam
(X_train,_),(_,_) = fashion_mnist.load_data()
X_train = (X_train.astype(np.float32)-127.5)/127.5
X_train = np.expand_dims(X_train, axis=-1)
latent_dim = 100
batch_size = 64
sample_interval = 2000
G = Sequential([
    Dense(128*7*7, input_shape=(latent_dim,), activation='relu'),
    Reshape((7,7,128)),
    Conv2DTranspose(128,4,strides=2,padding='same',activation='relu'),
    Conv2DTranspose(64,4,strides=2,padding='same',activation='relu'),
    Conv2DTranspose(1,7,activation='tanh',padding='same')
])
D = Sequential([
    Conv2D(64,3,strides=2,padding='same',input_shape=(28,28,1)),
    LeakyReLU(0.2),
    Conv2D(128,3,strides=2,padding='same'),
    LeakyReLU(0.2),
    Flatten(),
    Dense(1,activation='sigmoid')
])
D.compile(loss='binary_crossentropy',optimizer=Adam(0.0002,0.5),metrics=['accuracy'])
D.trainable = False
from keras.models import Model
from keras.layers import Input
z = Input(shape=(latent_dim,))
GAN = Model(z,D(G(z)))
GAN.compile(loss='binary_crossentropy',optimizer=Adam(0.0002,0.5))
valid = np.ones((batch_size,1))
fake = np.zeros((batch_size,1))
for e in range(10000):
    idx = np.random.randint(0,X_train.shape[0],batch_size)
    imgs = X_train[idx]
    noise = np.random.normal(0,1,(batch_size,latent_dim))
    gen_imgs = G.predict(noise,verbose=0)
    d_loss = 0.5*(D.train_on_batch(imgs,valid)[0]+D.train_on_batch(gen_imgs,fake)[0])
    noise = np.random.normal(0,1,(batch_size,latent_dim))
    g_loss = GAN.train_on_batch(noise,valid)
    if e % sample_interval == 0:
        print(f"{e} [D:{d_loss:.4f}] [G:{g_loss:.4f}]")
        s = np.random.normal(0,1,(9,latent_dim))
        g = G.predict(s,verbose=0); g = 0.5*g+0.5
        plt.figure(figsize=(3,3))
        for i in range(9):
            plt.subplot(3,3,i+1)
            plt.imshow(g[i,:,:,0],cmap='gray')
            plt.axis('off')
        plt.show()
        <img width="293" height="518" alt="504024991-fe0f9161-c40d-4b9b-8086-891d2853b73e" src="https://github.com/user-attachments/assets/1918c7a7-77e6-45eb-8ecf-c7d522e4de59" />
