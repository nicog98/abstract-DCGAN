# import necessary libraries
from PIL import Image, ImageOps
import numpy as np
import os
from glob import glob
import csv
from absl import flags, app

FLAGS = flags.FLAGS
# flags for extracting genre from wikiart
flags.DEFINE_boolean('get_artworks_by_genre', False, "Extract all artworks of a specific genre")
flags.DEFINE_integer('genre', 1, "Integer specifying which genre to extract")
flags.DEFINE_string('wikiart_from_path', '../../wikiart', "Path to wikiart directory")
# flags for augmenting images
flags.DEFINE_boolean('augment', False, "Augment all images specified in a directory")
# flags for resizing and saving images
flags.DEFINE_boolean('resize_and_save_images', False, "Resize and save all images in a directory")
# used for multiple settings
flags.DEFINE_string('from_path', None, "Path to dataset of images to augment")
flags.DEFINE_string('save_path', None, "Path to directory to save resized images")
flags.DEFINE_integer('output_size', None, "Resize images to this size")

def load_data(from_path, dataset, fname_pattern='*.jpg'):
	path_name = os.path.join(from_path, dataset, fname_pattern)
	imgs = glob(path_name)
	np.random.shuffle(imgs)
	return imgs

def get_images(image_files, output_size=128):
	images = []
	for fname in image_files:
		image = (Image.open(fname))
		width, height = image.size
		crop_size = min(width, height) # set the square crop side length to the minimum of width and height
		image = image.crop((0,0,crop_size, crop_size)) # crop image
		image = image.resize(size=(output_size, output_size), resample=Image.BILINEAR) # resize image
		images.append((np.asarray(image) - 127.5)/127.5)
	return np.array(images, dtype='float32')

# resize and save a singular image at file_path to save_path directory
def resize_and_save_image(file_path, save_path, output_size=None):
	filename = os.path.basename(file_path)
	print('processing', filename)
	image = Image.open(file_path)
	if output_size != None:
		image = image.resize((output_size, output_size))
	image.save(os.path.join(save_path, filename))

def get_artworks_by_genre(genre, database_path, csv_path, save_path, output_size=None):
	print('Database', database_path)
	print('CSV', csv_path)
	print('Save', save_path)
	# open csv file
	with open(csv_path) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		for row in csv_reader:
			# if this painting is part of the specified genre, resize and save in target directory
			if int(row[1]) == genre:
				resize_and_save_image(os.path.join(database_path, row[0]), save_path, output_size)

# load image
img_cnt = 0
img_trans = 0
def augment_data(from_path, save_path):
	global img_cnt
	global img_trans
	images = glob(os.path.join(from_path, '*.jpg'))
	for image in images:
		with open(image, 'rb') as file:
			img = Image.open(file)
			flip_and_mirror(img,128,save_path)
			img_rot_90 = img.rotate(90)
			flip_and_mirror(img_rot_90,128,save_path)
			img_cnt += 1
			img_trans = 0

def flip_and_mirror(img, size, save_path):
	do_cropping(img,size, save_path)  
	do_cropping(ImageOps.flip(img),size, save_path)  
	do_cropping(ImageOps.mirror(img),size, save_path)  
	do_cropping(ImageOps.flip(ImageOps.mirror(img)),size, save_path)  

def do_cropping(img, size, save_path):
	save(img, save_path)
	width, height = img.size
	img_size = min(width, height)
	loops = img_size // size
	for i in range(loops):
		for j in range(loops):
			cropped = img.crop((i*size, j*size, i*size+size, j*size+size))
			save(cropped, save_path)

def save(img, save_path):
	global img_trans
	img.save(save_path + '/'+ str(img_cnt) + '_' +str(img_trans) + '.jpg')
	img_trans += 1

# get all image files in directory specified by from_path and store resized images into to_path
def resize_and_save_images(from_path, save_path, fname_pattern="*.jpg", output_size=64):
	img_files = glob(os.join(from_path, fname_pattern))
	for filename in img_files:
		# resize image
		image = Image.open(filename)
		if output_size != None:
			image = image.resize((output_size, output_size))
		# save in new directory
		image.save(os.join(save_path, os.path.basename(filename)))

def transfer_images(image_names_path, abstract_path, save_path):
	image_names = glob(os.path.join(image_names_path, "*.jpg"))
	for image_name in image_names:
		filename = os.path.basename(image_name)
		image = Image.open(os.path.join(abstract_path, filename))
		image.save(os.path.join(save_path, filename))

def main(argv):
	if FLAGS.get_artworks_by_genre:
			get_artworks_by_genre(genre=FLAGS.genre, 
				database_path=os.path.join(FLAGS.wikiart_from_path,'wikiart'), 
				csv_path=os.path.join(FLAGS.wikiart_from_path,'wikiart_csv/genre_train.csv'), 
				save_path=FLAGS.save_path,
				output_size=FLAGS.output_size)
	
	if FLAGS.augment:
		augment_data(FLAGS.from_path, FLAGS.save_path)

	if FLAGS.resize_and_save_images:
		resize_and_save_images(FLAGS.from_path, FLAGS.save_path, output_size=FLAGS.output_size)

if __name__ == '__main__':
	app.run(main)