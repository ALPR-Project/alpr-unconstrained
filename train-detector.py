
import time
import sys
import numpy as np
import cv2
import argparse
from keras import backend as K
import keras
import tensorflow as tf
from random import choice
from os.path import isfile, isdir, basename, splitext
from os import makedirs

from src.keras_utils import save_model, load_model
from src.label import readShapes
from src.loss import loss
from src.utils import image_files_from_folder, show
from src.sampler import augment_sample, labels2output_map
from src.data_generator import DataGenerator
from src.evaluation_lpd import validar_lp_model
from keras.callbacks import TensorBoard

from pdb import set_trace as pause

def load_network(modelpath,input_dim):

	model = load_model(modelpath)
	input_shape = (input_dim,input_dim,3)

	# Fixed input size for training
	inputs  = keras.layers.Input(shape=(input_dim,input_dim,3))
	outputs = model(inputs)

	output_shape = tuple([s.value for s in outputs.shape[1:]])
	output_dim   = output_shape[1]
	model_stride = input_dim / output_dim

	assert input_dim % output_dim == 0, \
		'The output resolution must be divisible by the input resolution'

	assert model_stride == 2**4, \
		'Make sure your model generates a feature map with resolution ' \
		'16x smaller than the input'

	return model, model_stride, input_shape, output_shape

def process_data_item(data_item,dim,model_stride):
	image_ndarray = cv2.imread(data_item[0])
	XX,llp,pts = augment_sample(image_ndarray,data_item[1].pts,dim)
	YY = labels2output_map(llp,pts,dim,model_stride)
	return XX,YY


def salvar_imagem(imagem_array):
	ts = time.time()
	nome_arquivo = str(ts).replace('.', '')
	caminho_arquivo ='images_cropped/'+nome_arquivo+'.jpg'
	cv2.imwrite(caminho_arquivo, imagem_array)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-m' 		,'--model'			,type=str   , required=True		,help='Path to previous model')
	parser.add_argument('-n' 		,'--name'			,type=str   , required=True		,help='Model name')
	parser.add_argument('-tr'		,'--train-dir'		,type=str   , required=True		,help='Input data directory for training')
	parser.add_argument('-its'		,'--iterations'		,type=int   , default=300000	,help='Number of mini-batch iterations (default = 300.000)')
	parser.add_argument('-bs'		,'--batch-size'		,type=int   , default=32		,help='Mini-batch size (default = 32)')
	parser.add_argument('-od'		,'--output-dir'		,type=str   , default='./'		,help='Output directory (default = ./)')
	parser.add_argument('-op'		,'--optimizer'		,type=str   , default='Adam'	,help='Optmizer (default = Adam)')
	parser.add_argument('-lr'		,'--learning-rate'	,type=float , default=.01		,help='Optmizer (default = 0.01)')
	parser.add_argument('-vr' ,'--validate-dir' ,type=str, required=True ,help='Input data directory for validating')
	args = parser.parse_args()

	netname 	= basename(args.name)
	train_dir 	= args.train_dir
	outdir 		= args.output_dir
	validate_dir = args.validate_dir

	iterations 	= args.iterations
	batch_size 	= args.batch_size
	dim 		= 208

	if not isdir(outdir):
		makedirs(outdir)

	model,model_stride,xshape,yshape = load_network(args.model,dim)

	opt = getattr(keras.optimizers,args.optimizer)(lr=args.learning_rate)
	model.compile(loss=loss, optimizer=opt)

	print('Checking input directory...')
	Files = image_files_from_folder(train_dir)

	Data = []
	for file in Files:
		labfile = splitext(file)[0] + '.txt'
		if isfile(labfile):
			L = readShapes(labfile)
			# I = cv2.imread(file)
			Data.append([file,L[0]])

	print('%d images with labels found' % len(Data))

	dg = DataGenerator(	data=Data, \
						process_data_item_func=lambda x: process_data_item(x,dim,model_stride),\
						xshape=xshape, \
						yshape=(yshape[0],yshape[1],yshape[2]+1), \
						nthreads=2, \
						pool_size=500, \
						min_nsamples=100 )
	dg.start()
	print(' (after start) qtde de samples no buffer: %d ' % dg._count)
	Xtrain = np.empty((batch_size,dim,dim,3),dtype='single')
	Ytrain = np.empty((batch_size, int(dim / model_stride), int(dim / model_stride), 2 * 4 + 1))
	# Ytrain = np.empty((batch_size,dim/model_stride,dim/model_stride,2*4+1))

	model_path_backup = '%s/%s_backup' % (outdir,netname)
	model_path_final  = '%s/%s_final'  % (outdir,netname)

	# summary_writer = tf.summary.FileWriter('/media/jones/dataset/alpr/lotes_rotulacao/l1/logdir', sess.graph)
	summary_writer = tf.summary.FileWriter('/media/jones/dataset/alpr/fontes/lpr/wpod/logs2')

	# pylint: disable=maybe-no-member
	# summary.value.add(tag='validation_ds/accuracy', simple_value=accuracy_val)
	# summary_writer.add_summary(summary, step)
	total_loss_it = 0
	print(' (start iterating) qtde de samples no buffer: %d ' % dg._count)
	lr_ajustado = False
	for it in range(iterations):

		print('Iter. %d (of %d)' % (it+1,iterations))
		print("Learning rate: ", K.get_value(model.optimizer.lr))
		if not lr_ajustado and it > 0 and it % 100000 == 0:
			K.set_value(model.optimizer.lr, args.learning_rate / 10)
			lr_ajustado = True
		Xtrain,Ytrain = dg.get_batch(batch_size)
		print('qtde de samples no buffer: %d ' % dg._count)
		train_loss = model.train_on_batch(Xtrain,Ytrain)
		samples_imgs_denormalized = Xtrain * 255.
		samples_imgs_denormalized = samples_imgs_denormalized.astype('int')
		# for indice_image, image_sample_nd in enumerate(samples_imgs_denormalized):
		# 	print(str(indice_image))
		print('\tLoss: %f' % train_loss)
		total_loss_it += train_loss
		# Save model every 1000 iterations
		if (it+1) % 1000 == 0:
			mean_loss = total_loss_it/1000
			print('it %i , mean loss %f ' % (it, mean_loss))
			summary = tf.Summary()
			mAP_pascal, mAP_pascal_all_points, mAP_coco =  validar_lp_model(validate_dir,'/media/jones/dataset/alpr/lotes_rotulacao/validation_output', model)
			print('Saving model (%s)' % model_path_backup)
			summary.value.add(tag='train_loss', simple_value=mean_loss)
			summary.value.add(tag='mAP_pascal', simple_value=mAP_pascal)
			summary_writer.add_summary(summary, int(it + 1 / 100))
			save_model(model,model_path_backup)
			total_loss_it = 0

	print('Stopping data generator')
	dg.stop()

	print('Saving model (%s)' % model_path_final)
	save_model(model,model_path_final)
