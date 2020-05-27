# import necessary libraries
from PIL import Image
import numpy as np
import os
from glob import glob
import csv

# get all image files in directory specified by from_path and store resized images into to_path
def resize_and_save_images(from_path, save_path, fname_pattern="*.jpg", output_size=64):
	img_files = glob(os.join(from_path, fname_pattern))
	for filename in img_files:
		# resize image
		image = Image.open(filename)
		image = image.resize((output_size, output_size))
		# save in new directory
		image.save(os.join(save_path, os.path.basename(filename)))

# resize and save a singular image at file_path to save_path directory
def resize_and_save_image(file_path, save_path, output_size=64):
	filename = os.path.basename(file_path)
	print('processing', filename)
	image = Image.open(file_path)
	image = image.resize((output_size, output_size))
	image.save(os.path.join(save_path, filename))

def get_artworks_by_genre(genre, database_path, csv_path, save_path, output_size=64):
	print('Database', database_path)
	print('CSV', csv_path)
	print('Save', save_path)
	# open csv file
	with open(csv_path) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		for row in csv_reader:
			# if this painting is part of the specified genre, resize and save in target directory
			if int(row[1]) == genre:
				resize_and_save_image(os.path.join(database_path, row[0]), save_path)

def load_data(data_path, dataset_name, fname_pattern='*.jpg'):
	path_name = os.path.join(data_path, dataset_name, fname_pattern)
	return glob(path_name)

def get_images(image_files):
	images = []
	for fname in image_files:
		img_data = np.asarray(Image.open(fname))
		images.append((img_data - 127.5)/127.5)
	return np.array(images, dtype='float32')


# CODE FOR CREATING A FOLDER OF ALL PAINTINGS BELONGING TO A GENRE
# wikiart_data_path = '../../wikiart-data'

# get_artworks_by_genre(genre=4, 
# 	database_path=os.path.join(wikiart_data_path,'wikiart'), 
# 	csv_path=os.path.join(wikiart_data_path,'wikiart_csv/genre_train.csv'), 
# 	save_path=os.path.join(wikiart_data_path,'Landscapes'))