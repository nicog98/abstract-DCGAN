#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import layers, activations
import numpy as np
import matplotlib.pyplot as plt
import PIL
import os
import time
import util


# In[2]:


# Define layer dimensions for model
ngf = 64 # number of generative filters in first convolutional layer
ndf = 64 # number of discriminative filters in first convolutional layer
nc = 3 # number of channels in output (3 for RGB)
nz = 100 # number of dimensions for noise input to generator
# Hyperparameters
batch_size = 64
output_image_size = 64 # size of output images
lr = 2e-4
beta_1 = 0.5
num_samples = 64 # number of samples to generate at each epoch
epochs = 25
# logistics
save_every = 1
# Directories
data_path = "../"
dataset_name = "celeba-dataset"
checkpoint_dir = "./training_checkpoints"
trial_nr = 11 # which training trial this is
output_dir = "../images/trial_" + str(trial_nr)


# In[3]:

print("loading data...")
train_images = util.load_data(data_path, dataset_name)
print(len(train_images))

# In[4]:


# Create the models

# Generator
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(ngf*8*4*4, use_bias=False, input_shape=(nz,)))
    model.add(layers.Reshape((4, 4, ngf*8)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    assert model.output_shape == (None, 4, 4, ngf*8) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(ngf*4, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, ngf*4)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(ngf*2, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, ngf*2)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    model.add(layers.Conv2DTranspose(ngf*1, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, ngf)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(nc, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 64, 64, nc)

    return model


# In[5]:


# use the untrained generator to generate an image
print("creating generator...")
generator = make_generator_model()
noise = tf.random.normal([1,nz])
generated_image = generator(noise, training=False)
print(generated_image.shape)
# plt.imshow((generated_image.numpy()[0, :, :, :] * 127.5 + 127.5).astype(int))


# In[6]:


# Discriminator
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(ndf, (5, 5), strides=(2, 2), padding='same', input_shape=[64, 64, nc]))
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None, 32, 32, ndf)

    model.add(layers.Conv2D(ndf*2, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None, 16, 16, ndf*2)

    model.add(layers.Conv2D(ndf*4, (5, 5), strides=(2, 2), padding='same'))
    print(model.output_shape)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None, 8, 8, ndf*4)

    model.add(layers.Conv2D(ndf*8, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None, 4, 4, ndf*8)
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


# In[7]:

print("creating discriminator...")
discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print(decision)


# In[8]:


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# In[9]:


# Define losses
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1)


# In[10]:


# Save checkpoints
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


# In[11]:


# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_samples, nz])


# In[12]:


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, nz])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        batch_idxs = len(dataset) // batch_size # number of batches in train dataset
        for idx in range(batch_idxs):
            step_time = time.time()
            image_batch = util.get_images(dataset[idx*batch_size:(idx+1)*batch_size])
            gen_loss, disc_loss = train_step(image_batch)
            print("Epoch: [{}/{}] [{}/{}] took: {:.3f}, d_loss: {:.5f}, g_loss: {:.5f}".format(
                epoch+1, epochs, idx, batch_idxs, time.time()-step_time, disc_loss, gen_loss))

        # Produce images for the GIF as we go
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed)

        # Save the model every 15 epochs
        if (epoch + 1) % save_every == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    generate_and_save_images(generator,
                           epochs,
                           seed)


# In[13]:


# Generate and save images
def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(8,8))

    for i in range(predictions.shape[0]):
        plt.subplot(8, 8, i+1)
        plt.imshow((predictions.numpy()[i, :, :, :] * 127.5 + 127.5).astype(int)) # put image on [0,255] range
        plt.axis('off')

    plt.savefig(output_dir + '/' + 'image_at_epoch_{:04d}.png'.format(epoch))
    plt.clf()


# In[14]:


# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).assert_consumed()


# In[ ]:


# Train the model
train(train_images, epochs)


# In[ ]:




