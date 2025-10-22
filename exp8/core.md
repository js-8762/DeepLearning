from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU
from keras.optimizers import Adam
import numpy as np
from keras.datasets import mnist

def build_generator():
    return Sequential([
        Dense(128*7*7, activation="relu", input_dim=100),
        Reshape((7,7,128)),
        Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'),
        LeakyReLU(0.2),
        Conv2DTranspose(1, kernel_size=4, strides=2, padding='same', activation='tanh')
    ])

def build_discriminator():
    return Sequential([
        Conv2D(64, kernel_size=3, strides=2, input_shape=(28,28,1), padding='same'),
        LeakyReLU(0.2),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])

def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential([generator, discriminator])
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return model

def train(epochs, batch_size=64, print_interval=100):
    (X_train, _), (_, _) = mnist.load_data()
    X_train = (X_train.astype(np.float32)-127.5)/127.5
    X_train = np.expand_dims(X_train, axis=-1)
    half_batch = batch_size // 2

    for epoch in range(1, epochs+1):
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        real_imgs = X_train[idx]
        noise = np.random.normal(0,1,(half_batch,100))
        fake_imgs = generator.predict(noise, verbose=0)
        d_loss = 0.5 * (discriminator.train_on_batch(real_imgs, np.ones((half_batch,1))) +
                        discriminator.train_on_batch(fake_imgs, np.zeros((half_batch,1))))
        g_loss = gan.train_on_batch(np.random.normal(0,1,(batch_size,100)), np.ones((batch_size,1)))

        if epoch % print_interval == 0 or epoch==1 or epoch==epochs:
            print(f"Epoch {epoch}/{epochs} | D loss: {d_loss:.4f} | G loss: {g_loss:.4f}")

generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002,0.5))
gan = build_gan(generator, discriminator)
train(epochs=1000, batch_size=64, print_interval=100)
<img width="1456" height="314" alt="501270122-f5734611-f007-48bc-8add-993ed91ccb3e" src="https://github.com/user-attachments/assets/53a68ab0-bcf8-4359-92ff-ba8d700d2fdf" />
