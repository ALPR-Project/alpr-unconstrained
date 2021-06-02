
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


def calc_iou(gt_bbox, pred_bbox):
	'''
	This function takes the predicted bounding box and ground truth bounding box and
	return the IoU ratio
	'''
	x_topleft_gt, y_topleft_gt, x_bottomright_gt, y_bottomright_gt = gt_bbox
	x_topleft_p, y_topleft_p, x_bottomright_p, y_bottomright_p = pred_bbox
	if (x_topleft_gt > x_bottomright_gt) or (y_topleft_gt > y_bottomright_gt):
		raise AssertionError("Ground Truth Bounding Box is not correct")
	if (x_topleft_p > x_bottomright_p) or (y_topleft_p > y_bottomright_p):
		raise AssertionError("Predicted Bounding Box is not correct", x_topleft_p, x_bottomright_p, y_topleft_p, y_bottomright_gt)
	# if the GT bbox and predcited BBox do not overlap then iou=0
	if x_bottomright_gt < x_topleft_p:
		# If bottom right of x-coordinate  GT  bbox is less than or above the top left of x coordinate of  the predicted BBox
		return 0.0
	if y_bottomright_gt < y_topleft_p:  # If bottom right of y-coordinate  GT  bbox is less than or above the top left of y coordinate of  the predicted BBox
		return 0.0
	if x_topleft_gt > x_bottomright_p:  # If bottom right of x-coordinate  GT  bbox is greater than or below the bottom right  of x coordinate of  the predcited BBox
		return 0.0
	if y_topleft_gt > y_bottomright_p:  # If bottom right of y-coordinate  GT  bbox is greater than or below the bottom right  of y coordinate of  the predcited BBox
		return 0.0
	GT_bbox_area = (x_bottomright_gt - x_topleft_gt + 1) * (y_bottomright_gt - y_topleft_gt + 1)
	Pred_bbox_area = (x_bottomright_p - x_topleft_p + 1) * (y_bottomright_p - y_topleft_p + 1)
	x_top_left = np.max([x_topleft_gt, x_topleft_p])
	y_top_left = np.max([y_topleft_gt, y_topleft_p])
	x_bottom_right = np.min([x_bottomright_gt, x_bottomright_p])
	y_bottom_right = np.min([y_bottomright_gt, y_bottomright_p])
	intersection_area = (x_bottom_right - x_top_left + 1) * (y_bottom_right - y_top_left + 1)
	union_area = (GT_bbox_area + Pred_bbox_area - intersection_area)
	return intersection_area / union_area


def validar_lp_model(entrada_diretorio_validacao, diretorio_saida, wpod_net):
	print('iniciando validacao modelo')
	lp_threshold = .4
	imgs_paths = glob('%s/*.jpg' % entrada_diretorio_validacao)
	print('Searching for license plates using WPOD-NET')
	for i,img_path in enumerate(imgs_paths):
		# print('\t Processing %s' % img_path)
		bname_image_file = splitext(basename(img_path))[0]
		name_file_gt = bname_image_file+'.txt'
		name_file = basename(img_path)
		Ivehicle = cv2.imread(img_path)
		height, width = Ivehicle.shape[:2]
		ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
		side  = int(ratio*288.)
		bound_dim = min(side + (side%(2**4)),608)
		# print("\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio))
		Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,(240,80),lp_threshold)
		lista_preds_frame = []
		gt_img_path = img_path.replace('.jpg', '.txt')
		ground_truth_frame = []
		with open(gt_img_path) as f:
			lines = f.readlines()
			for linha in lines:
				pontos = linha.split(',')[1:9]
				pontos = [int(float(ponto)*width) if indice < 4 else int(float(ponto)*height)  for indice, ponto in enumerate(pontos)]
				# ground_truth = [int(float(pontos[0]) * width), int(float(pontos[4]) * height),
				# 				int(float(pontos[2]) * width), int(float(pontos[6]) * height), 0, 0, 0]
				x_points = pontos[0:4] * width
				y_points = pontos[4:8] * height
				# top_left_plate = int(float(pontos[0]) * width), int(float(pontos[4]) * height)
				top_left_plate_x = min(x_points)
				top_left_plate_y = min(y_points)
				bottom_right_plate_x = max(x_points)
				bottom_right_plate_y = max(y_points)
				top_left_plate = top_left_plate_x, top_left_plate_y
				# bottom_right_plate = int(float(pontos[2]) * width), int(float(pontos[6]) * height)
				bottom_right_plate = bottom_right_plate_x, bottom_right_plate_y
				# gt = [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
				ground_truth_frame.append([top_left_plate[0], top_left_plate[1], bottom_right_plate[0], bottom_right_plate[1], 0, 0, 0])
		if len(LlpImgs):
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
		else:
			print('imagem sem placa detectacada %s ' % img_path)
			lista_preds_frame.append([0, 0, 0, 0, 0, 0])
			# metric_fn.add(np.array(lista_preds_frame), np.array([[0, 0, 0, 0, 0, 0, 0]]))
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
