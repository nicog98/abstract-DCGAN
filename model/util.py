# import necessary libraries
from PIL import Image
import matplotlib
from matplotlib import image, pyplot
import numpy as np
from os import listdir

# get all image files in directory specified by from_path and store resized images into to_path
def resize_and_save_images(from_path, to_path):
	for filename in listdir(from_path):
		if filename.endswith(".jpg"):
			# resize image
			image = Image.open(from_path + '/' + filename)
			image_resized = image.resize((64, 64))
			# save in new directory
			image_resized.save(to_path + '/' + filename)

def load_data(data_path):
	images = []
	for filename in listdir(data_path):
        # skip hidden files
		if filename.endswith(".jpg"):
			img_data = image.imread(data_path + '/' + filename)
			images.append(img_data)
	data = np.array(images).astype('float32')
	data = (data - 127.5) / 127.5 # normalize images on [-1,1]
	return data