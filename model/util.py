# import necessary libraries
from PIL import Image
import numpy as np
import os
from glob import glob

# get all image files in directory specified by from_path and store resized images into to_path
def resize_and_save_images(from_path, to_path):
	for filename in listdir(from_path):
		if filename.endswith(".jpg"):
			# resize image
			image = Image.open(from_path + '/' + filename)
			image_resized = image.resize((64, 64))
			# save in new directory
			image_resized.save(to_path + '/' + filename)

def load_data(data_path, dataset_name, fname_pattern='*.jpg'):
	path_name = os.path.join(data_path, dataset_name, fname_pattern)
	return glob(path_name)

def get_images(image_files):
	images = []
	for fname in image_files:
		img_data = np.asarray(Image.open(fname))
		images.append((img_data - 127.5)/127.5)
	return np.array(images, dtype='float32')

filenames = load_data('../', 'test-dataset')
images = get_images(filenames)
print(images.shape)
print(images[0])