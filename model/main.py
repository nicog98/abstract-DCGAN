import tensorflow as tf
from tensorflow.keras import layers, activations
import numpy as np
import matplotlib.pyplot as plt
import PIL
import os
from time import time, ctime
import util

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
# Define layer dimensions for model
flags.DEFINE_integer("ngf", 150, "number of generative filters in first convolutional layer")
flags.DEFINE_integer("ndf", 50, "number of discriminative filters in first convolutional layer")
flags.DEFINE_integer("nc", 3, "number of channels in output (3 for RGB image)")
flags.DEFINE_integer("nz", 100, "number of dimensions for noise input to generator")
# Hyperparameters
flags.DEFINE_integer("batch_size", 64, "number of images in each training batch")
flags.DEFINE_integer("output_image_size", 128, "size of output images")
flags.DEFINE_float("lr", 2e-4, "learning rate")
flags.DEFINE_float("beta_1", 0.5, "beta 1 for Adam optimizer")
flags.DEFINE_float("alpha", 0.2, "alpha for LeakyReLU in Discriminator")
flags.DEFINE_integer("num_samples", 64, "number of samples to generate after each epoch")
flags.DEFINE_integer("start_epoch", 0, "Epoch to start from (use if restoring from checkpoint to continue training.")
flags.DEFINE_integer("epochs", 100, "number of epochs to train for")
# logistics for training
flags.DEFINE_bool("train", False, "Whether or not to train the model")
flags.DEFINE_bool("generate", False, "If true, use the model to generate num_samples images")
flags.DEFINE_bool("restore_from_checkpoint", False, "determine whether or not to restore from the most recent checkpoint")
flags.DEFINE_integer("save_every", 25, "save training checkpoints every number of epochs")
# Directories
flags.DEFINE_string("data_path", "../", "Path to the directory where the dataset is stored")
flags.DEFINE_string("dataset", "Abstract", "Name of the dataset to be used")
flags.DEFINE_string("checkpoint_dir", "./training_checkpoints", "Path to directory to store training checkpoints")
flags.DEFINE_string("output_dir", "../images/", "Path to directory to store sample outputs after every epoch")
flags.DEFINE_string("generated_dir", None, "Specify the directory for images generated from model.")

# Generator
def make_generator_model():
	# Number of filters, input and output dimensions
	ngf = FLAGS.ngf
	nz = FLAGS.nz
	nc = FLAGS.nc
	# Reshape layer (nz,) -> (4,4,ngf*16)
	model = tf.keras.Sequential()
	model.add(layers.Dense(ngf*16*4*4, use_bias=False, input_shape=(nz,)))
	model.add(layers.Reshape((4, 4, ngf*16)))
	model.add(layers.BatchNormalization())
	model.add(layers.ReLU())
	assert model.output_shape == (None, 4, 4, ngf*16) # Note: None is the batch size
	# First convolutional layer: (4,4,ngf*16) -> (8,8, ngf*8)
	model.add(layers.Conv2DTranspose(ngf*8, (4, 4), strides=(2, 2), padding='same', use_bias=False))
	assert model.output_shape == (None, 8, 8, ngf*8)
	model.add(layers.BatchNormalization())
	model.add(layers.ReLU())
	# Second convolutional layer: (8,8,ngf*8) -> (16,16,ngf*4)
	model.add(layers.Conv2DTranspose(ngf*4, (4, 4), strides=(2, 2), padding='same', use_bias=False))
	assert model.output_shape == (None, 16, 16, ngf*4)
	model.add(layers.BatchNormalization())
	model.add(layers.ReLU())
	# Third convolutional layer: (16, 16, ngf*4) -> (32, 32, ngf*2)
	model.add(layers.Conv2DTranspose(ngf*2, (4, 4), strides=(2, 2), padding='same', use_bias=False))
	assert model.output_shape == (None, 32, 32, ngf*2)
	model.add(layers.BatchNormalization())
	model.add(layers.ReLU())
	# Fourth convolutional layer: (32, 32, ngf*2) -> (64, 64, ngf)
	model.add(layers.Conv2DTranspose(ngf, (4, 4), strides=(2, 2), padding='same', use_bias=False))
	assert model.output_shape == (None, 64, 64, ngf)
	model.add(layers.BatchNormalization())
	model.add(layers.ReLU())
	# Fifth convolutional layer: (64, 64, ngf) -> (128, 128, nc)
	model.add(layers.Conv2DTranspose(nc, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
	assert model.output_shape == (None, 128, 128, nc)

	return model

# Discriminator
def make_discriminator_model():
	ndf = FLAGS.ndf
	nc = FLAGS.nc
	model = tf.keras.Sequential()
	model.add(layers.Conv2D(ndf, (4, 4), strides=(2, 2), padding='same', input_shape=[128, 128, nc]))
	model.add(layers.LeakyReLU(alpha=FLAGS.alpha))
	assert model.output_shape == (None, 64, 64, ndf)

	model.add(layers.Conv2D(ndf*2, (4, 4), strides=(2, 2), padding='same'))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU(alpha=FLAGS.alpha))
	assert model.output_shape == (None, 32, 32, ndf*2)

	model.add(layers.Conv2D(ndf*4, (4, 4), strides=(2, 2), padding='same'))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU(alpha=FLAGS.alpha))
	assert model.output_shape == (None, 16, 16, ndf*4)

	model.add(layers.Conv2D(ndf*8, (4, 4), strides=(2, 2), padding='same'))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU(alpha=FLAGS.alpha))
	assert model.output_shape == (None, 8, 8, ndf*8)

	model.add(layers.Conv2D(ndf*16, (4, 4), strides=(2, 2), padding='same'))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU(alpha=FLAGS.alpha))
	assert model.output_shape == (None, 4, 4, ndf*16)
	
	model.add(layers.Flatten())
	model.add(layers.Dense(1))

	return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Define losses
def discriminator_loss(real_output, fake_output):
	real_loss = cross_entropy(tf.ones_like(real_output), real_output)
	fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
	total_loss = real_loss + fake_loss
	return total_loss

def generator_loss(fake_output):
	return cross_entropy(tf.ones_like(fake_output), fake_output)

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(FLAGS.lr, FLAGS.beta_1)
discriminator_optimizer = tf.keras.optimizers.Adam(FLAGS.lr, FLAGS.beta_1)

# We will reuse this noise overtime (so it's easier)
# to visualize progress in the animated GIF)
noise = tf.random.normal([FLAGS.num_samples, FLAGS.nz])

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
	noise = tf.random.normal([FLAGS.batch_size, FLAGS.nz])

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

def train(dataset, start_epoch=0, epochs=100):
	batch_size = FLAGS.batch_size
	for epoch in range(start_epoch, epochs):
		start = time()
		batch_idxs = len(dataset) // batch_size # number of batches in train dataset
		for idx in range(batch_idxs):
			step_time = time()
			image_batch = util.get_images(dataset[idx*batch_size:(idx+1)*batch_size])
			gen_loss, disc_loss = train_step(image_batch)
			print("Epoch: [{}/{}] [{}/{}] took: {:.3f}, d_loss: {:.5f}, g_loss: {:.5f}".format(
				epoch+1, epochs, idx+1, batch_idxs, time()-step_time, disc_loss, gen_loss))

		# Produce images for the GIF as we go
		generate_and_save_images(generator,
							   noise,
							   output_dir=FLAGS.output_dir,
							   image_name='image_at_epoch_{:04d}.png'.format(epoch+1))

		# Save the model after every number of epochs
		if (epoch+1) % FLAGS.save_every == 0:
			print("saving checkpoint at", epoch+1)
			checkpoint.save(file_prefix = checkpoint_prefix)

		print ('Time for epoch {} is {} sec'.format(epoch + 1, time()-start))

	# Generate after the final epoch
	generate_and_save_images(generator,
						   noise,
						   output_dir=FLAGS.output_dir,
						   image_name='image_at_epoch_{:04d}.png'.format(epoch+1))


# Generate and save images
def generate_and_save_images(model, noise, output_dir, image_name):
	# Notice `training` is set to False.
	# This is so all layers run in inference mode (batchnorm).
	predictions = model(noise, training=False)
	print(predictions.shape)

	fig = plt.figure(figsize=(8,8))

	for i in range(predictions.shape[0]):
		plt.subplot(8, 8, i+1)
		plt.imshow((predictions.numpy()[i, :, :, :] * 127.5 + 127.5).astype(int)) # put image on [0,255] range
		plt.axis('off')

	# make the directory if it is not already present
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	plt.savefig(output_dir + '/' + image_name)
	plt.clf()

if __name__ == '__main__':

	# load the data
	if FLAGS.train:
		print("loading data from {} dataset...".format(FLAGS.dataset))
		train_images = util.load_data(FLAGS.data_path, FLAGS.dataset)
		print(len(train_images), "images loaded.")

	# Create the generator and discriminator
	print("creating generator...")
	test_input = tf.random.normal([1,FLAGS.nz])
	generator = make_generator_model()
	generated_image = generator(test_input, training=False) # use the untrained generator to generate an image
	print(generated_image.shape)

	print("creating discriminator...")
	discriminator = make_discriminator_model()
	decision = discriminator(generated_image) # get the untrained discriminator decision on generated image
	print(decision)

	# Save checkpoints
	checkpoint_prefix = os.path.join(FLAGS.checkpoint_dir, "ckpt")
	checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
									 discriminator_optimizer=discriminator_optimizer,
									 generator=generator,
									 discriminator=discriminator)

	# Train the model
	if FLAGS.restore_from_checkpoint: # restore from the latest chec
		checkpoint.restore(tf.train.latest_checkpoint(FLAGS.checkpoint_dir))

	if FLAGS.train:
		train(train_images, FLAGS.start_epoch, FLAGS.epochs)

	if FLAGS.generate:
		if FLAGS.generated_dir == None:
			raise Exception("generated_dir flag not defined.")
		generate_and_save_images(generator, noise, FLAGS.generated_dir, image_name="Generated Image {}.png".format(ctime()))



