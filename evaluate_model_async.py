
import argparse
import sys, os
import keras
import cv2
import numpy as np
import traceback
import xml.etree.ElementTree as ET
from src.keras_utils 			import load_model
from glob 						import glob
from os.path 					import splitext, basename
from src.utils 					import im2single
from src.keras_utils 			import load_model, detect_lp
from src.label 					import Shape, writeShapes
from mean_average_precision import MetricBuilder
print(MetricBuilder.get_metrics_list())

metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=1)

is_exibir_gt = True
bbox_color_gt = (0, 0, 255)





def validate_model(wpod_net_path, validate_dir, output_dir):
	wpod_net = load_model(wpod_net_path)
	validar_lp_model(validate_dir, output_dir, wpod_net)




def validar_lp_model(entrada_diretorio_validacao, diretorio_saida, wpod_net):
	print('iniciando validacao modelo')
	lp_threshold = .4
	imgs_paths = glob('%s/*.jpg' % entrada_diretorio_validacao)
	print('Searching for license plates using WPOD-NET')
	for i,img_path in enumerate(imgs_paths):
		print('\t Processing %s' % img_path)
		bname_image_file = splitext(basename(img_path))[0]
		name_file_gt = bname_image_file+'.txt'
		name_file = basename(img_path)
		Ivehicle = cv2.imread(img_path)
		height, width = Ivehicle.shape[:2]
		ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
		side  = int(ratio*288.)
		bound_dim = min(side + (side%(2**4)),608)
		print("\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio))
		Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,(240,80),lp_threshold)
		if len(LlpImgs):
			lista_preds_frame = []
			for indice_bbox,Ilp in enumerate(LlpImgs):
				# Ilp = LlpImgs[0]
				Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
				Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
				license_plate_bouding_box_relativo = Llp[indice_bbox]
				probability = license_plate_bouding_box_relativo._Label__prob
				top_left_rl_tp = tuple(license_plate_bouding_box_relativo._Label__tl.tolist())
				top_left_x = int(top_left_rl_tp[0]*width)
				top_left_y = int(top_left_rl_tp[1] * height)
				bottom_right_rl = license_plate_bouding_box_relativo._Label__br.tolist()
				bottom_right_x = int(bottom_right_rl[0]*width)
				bottom_right_y = int(bottom_right_rl[1]* height)
				plate_width = bottom_right_x  - top_left_x
				plate_height = bottom_right_y  - top_left_y
				red_color = (255, 0, 0)
				bbox_color = red_color
				# car_rgb_np = cv2.cvtColor(Ivehicle, cv2.COLOR_BGR2RGB)
				car_rgb_np = Ivehicle
				thickness = 1
				# [xmin, ymin, xmax, ymax, class_id, confidence]
				if top_left_x >0 and top_left_y > 0 and bottom_right_x > 0 and bottom_right_y > 0:
					preds_frame = [int(top_left_x), int(top_left_y), int(bottom_right_x), int(bottom_right_y), 0, probability]
					lista_preds_frame.append(preds_frame)
				cv2.rectangle(car_rgb_np, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), bbox_color, thickness)  # filled
				s = Shape(Llp[0].pts)
				# cv2.imwrite('%s/%s_lp_car.png' % (output_dir,bname), car_rgb_np)
				# cv2.imwrite('%s/%s_lp.png' % (output_dir,bname),Ilp*255.)
				# writeShapes('%s/%s_lp.txt' % (diretorio_saida,bname_image_file),[s])
			gt_img_path = img_path.replace('.jpg', '.txt')
			ground_truth_frame = []
			with open(gt_img_path) as f:
				lines = f.readlines()
				for linha in lines:
					pontos = linha.split(',')[1:9]
					ground_truth = [int(float(pontos[0])*width), int(float(pontos[4])*height), int(float(pontos[2])*width), int(float(pontos[6])*height), 0, 0, 0]
					ground_truth_frame.append(ground_truth)
			metric_fn.add(np.array(lista_preds_frame), np.array(ground_truth_frame))

		if is_exibir_gt and len(LlpImgs):
			for bbox_gt in ground_truth_frame:
				cv2.rectangle(car_rgb_np, (bbox_gt[0], bbox_gt[1]), (bbox_gt[2], bbox_gt[3]), bbox_color_gt, thickness)  # filled
			cv2.imwrite('%s/%s_lp_car.png' % (diretorio_saida, bname_image_file), car_rgb_np)
	print('calculando mAP')
	# compute PASCAL VOC metric
	mAP_pascal = metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']
	print('VOC PASCAL mAP:'+str(mAP_pascal))
	# compute PASCAL VOC metric at the all points
	mAP_pascal_all_points = metric_fn.value(iou_thresholds=0.5)['mAP']
	print('VOC PASCAL mAP in all points:'+str(mAP_pascal_all_points))
	# compute metric COCO metric
	mAP_coco = metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
	print('COCO mAP: '+str(mAP_coco))
	return mAP_pascal, mAP_pascal_all_points, mAP_coco


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-m' 		,'--model'			,type=str   , required=True		,help='Path to previous model')
	parser.add_argument('-vr'		,'--validate-dir'		,type=str   , required=True		,help='Input data directory for training')
	parser.add_argument('-o' ,'--output-dir' ,type=str, required=True ,help='Input data directory for training')
	args = parser.parse_args()
	wpod_net_path = args.model
	validate_dir = args.validate_dir
	output_dir = args.output_dir
	validate_model(wpod_net_path, validate_dir, output_dir)
