# Import necessary libraries
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Define the generator model
generator = keras.Sequential([
    layers.Dense(128, input_dim=100, activation='relu'),
    layers.Dense(784, activation='sigmoid'),
    layers.Reshape((28, 28))
])

# Define the discriminator model
discriminator = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the discriminator
discriminator.compile(optimizer='adam', loss='binary_crossentropy')

# Combine generator and discriminator to create GAN
discriminator.trainable = False
gan_input = keras.Input(shape=(100,))
gan_output = discriminator(generator(gan_input))
gan = keras.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Load dataset (e.g., MNIST)
(x_train, _), (_, _) = keras.datasets.mnist.load_data()

# Normalize data
x_train = x_train.astype('float32') / 255

# Reshape data
x_train = x_train.reshape(-1, 28, 28)

# Training loop
epochs = 100
batch_size = 64

for epoch in range(epochs):
    for batch in range(x_train.shape[0] // batch_size):
        # Train discriminator
        noise = np.random.normal(0, 1, size=[batch_size, 100])
        generated_images = generator.predict(noise)
        real_images = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]
        X = np.concatenate([real_images, generated_images])
        y_dis = np.zeros(2*batch_size)
        y_dis[:batch_size] = 0.9  # label smoothing
        discriminator.trainable = True
        discriminator.train_on_batch(X, y_dis)

        # Train generator
        noise = np.random.normal(0, 1, size=[batch_size, 100])
        y_gen = np.ones(batch_size)
        discriminator.trainable = False
        gan.train_on_batch(noise, y_gen)

    print(f'Epoch: {epoch+1}, Discriminator Loss: {discriminator_loss}, Generator Loss: {generator_loss}')

# Generate some images
generated_images = generator.predict(np.random.normal(0, 1, size=[10, 100]))

# Display generated images
import matplotlib.pyplot as plt
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(generated_images[i], cmap='gray')
    plt.axis('off')
plt.show()
