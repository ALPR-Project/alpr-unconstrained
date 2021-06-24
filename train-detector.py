import os
from glob import glob
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
from evaluate_model_async import validar_lp_model
from keras.callbacks import TensorBoard

from pdb import set_trace as pause

def load_network(modelpath,input_dim):

	model = load_model(modelpath)
	input_shape = (input_dim,input_dim,3)

	# Fixed input size for training
	inputs = keras.layers.Input(shape=(input_dim,input_dim,3))
	outputs = model(inputs)

	output_shape = tuple([s.value for s in outputs.shape[1:]])
	output_dim = output_shape[1]
	model_stride = input_dim / output_dim

	assert input_dim % output_dim == 0, \
		'The output resolution must be divisible by the input resolution'

	assert model_stride == 2**4, \
		'Make sure your model generates a feature map with resolution ' \
		'16x smaller than the input'

	return model, model_stride, input_shape, output_shape

def process_data_item(data_item,dim_w, dim_h,model_stride):
	image_ndarray = cv2.imread(data_item[0])
	XX,llp,pts_transformed = augment_sample(image_ndarray,data_item[1].pts,dim_w, dim_h)
	YY = labels2output_map(llp,pts_transformed, dim_w, dim_h,model_stride)
	return XX, YY, pts_transformed


def salvar_imagem(imagem_array, caminho_completo_arquivo):
	cv2.imwrite(caminho_completo_arquivo, imagem_array)



if __name__ == '__teste__':
	data_item = []
	# L = readShapes('/media/jones/dataset/alpr/lotes_rotulacao/preprocessados/train/lote1_1703_2000000544.txt')
	# file = '/media/jones/dataset/alpr/lotes_rotulacao/preprocessados/train/lote1_1703_2000000544.jpg'
	L = readShapes('l4_160_3104_1_1_000000446.txt')
	file = 'l4_160_3104_1_1_000000446.jpg'
	# I = cv2.imread(file)
	dim_w = 304
	dim_h = 304
	# data_item.append([file, L[0]])
	data_item.append(file)
	data_item.append(L[0])
	model_stride = 16.0
	XX,YY_output_map, pts_transformed = process_data_item(data_item, dim_w, dim_h, model_stride)
	file_gr_transformed = open('samples/l4_160_3104_1_1_000000446_transf.txt', 'w')
	file_gr_transformed.write(str(pts_transformed))
	file_gr_transformed.close()
	img_ndarray = XX*255
	ts = time.time()
	nome_arquivo = str(ts).replace('.', '')
	caminho_arquivo = 'samples/' + nome_arquivo + '.jpg'
	salvar_imagem(img_ndarray, caminho_arquivo)
	caminho_train = '/media/jones/datarec/lpr/dataset/versao_atual/preprocessados/train_moto'
	imgs_paths = glob('%s/*.jpg' % caminho_train)
	diretorio_saida = '/media/jones/datarec/lpr/dataset/versao_atual/preprocessados/samples_train_moto'
	if not os.path.exists(diretorio_saida):
		os.makedirs(diretorio_saida)
	print('Searching for license plates using WPOD-NET')
	for i,img_path in enumerate(imgs_paths):
		Ivehicle = cv2.imread(img_path)
		gt_img_path = img_path.replace('.jpg', '.txt')
		data_item = []
		L = readShapes(gt_img_path)
		dim_w = 208
		dim_h = 208
		# data_item.append([file, L[0]])
		data_item.append(img_path)
		data_item.append(L[0])
		model_stride = 16.0
		XX, YY_output_map, pts_transformed = process_data_item(data_item, dim_w, dim_h, model_stride)
		img_ndarray = XX * 255
		nome_arquivo = os.path.basename(img_path)
		caminho_completo_saida = os.path.abspath(os.path.join(diretorio_saida, nome_arquivo))
		salvar_imagem(img_ndarray, caminho_completo_saida)
	# data_item.append()


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
	parser.add_argument('-ld' ,'--logdir' ,type=str, required=True ,help='Input data directory for validating')
	parser.add_argument('-vod' ,'--validation_output_dir' ,type=str, required=True ,help='Input data directory for validating')
	parser.add_argument('-me' ,'--modo_execucao' , type=str, help='0 pra treino. 1 pra teste.', default='0')
	args = parser.parse_args()

	netname 	= basename(args.name)
	train_dir 	= args.train_dir
	outdir 		= args.output_dir
	validate_dir = args.validate_dir
	logdir = args.logdir
	validation_output_dir = args.validation_output_dir


	iterations 	= args.iterations
	batch_size 	= args.batch_size
	dim  = 208
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
						process_data_item_func=lambda x: process_data_item(x,dim, dim,model_stride),\
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
	summary_writer = tf.summary.FileWriter(logdir)

	# pylint: disable=maybe-no-member
	# summary.value.add(tag='validation_ds/accuracy', simple_value=accuracy_val)
	# summary_writer.add_summary(summary, step)
	total_loss_it = 0
	print(' (start iterating) qtde de samples no buffer: %d ' % dg._count)
	lr_ajustado = False
	qtd_iterations_per_epoch = 1000
	for it in range(iterations):

		print('Iter. %d (of %d)' % (it+1,iterations))
		print("Learning rate: ", K.get_value(model.optimizer.lr))
		if not lr_ajustado and it > 0 and it % 150000 == 0:
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
		if (it+1) % qtd_iterations_per_epoch == 0:
			mean_loss = total_loss_it/qtd_iterations_per_epoch
			print('it %i , mean loss %f ' % (it, mean_loss))
			summary = tf.Summary()
			mAP_pascal, mAP_pascal_all_points, mAP_coco, indicadores_validacao = validar_lp_model(validate_dir, validation_output_dir, model)
			print('Saving model (%s)' % model_path_backup)
			summary.value.add(tag='train_loss', simple_value=mean_loss)
			summary.value.add(tag='mAP_pascal', simple_value=mAP_pascal)
			precision_car, recall_car = indicadores_validacao.precision_recall_car()
			summary.value.add(tag='precision/precision car', simple_value=precision_car)
			summary.value.add(tag='recall/recall car', simple_value=recall_car)
			summary.value.add(tag='fn/false negative car', simple_value=indicadores_validacao.false_negative_car)
			precision_moto, recall_moto = indicadores_validacao.precision_recall_moto()
			summary.value.add(tag='precision/precision moto', simple_value=precision_moto)
			summary.value.add(tag='recall/recall moto', simple_value=recall_moto)
			summary.value.add(tag='fn/false negative moto', simple_value=indicadores_validacao.false_negative_moto)
			summary_writer.add_summary(summary, int(it + 1 / 100))
			save_model(model,model_path_backup)
			total_loss_it = 0

	print('Stopping data generator')
	dg.stop()

	print('Saving model (%s)' % model_path_final)
	save_model(model,model_path_final)
