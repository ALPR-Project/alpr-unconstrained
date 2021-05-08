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

def adjust_pts(pts,lroi):
	return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))



def point_str_to_tuple(point_str):
	pontos = point_str.split(',')
	return int(float(pontos[0])), int(float(pontos[1]))


def get_pontos_extremos(pontos):
	ponto_zero =  np.array([0, 0])
	menor_distancia = 10000
	maior_distancia = 0
	indice_menor_distancia = None
	indice_maior_distancia = None
	for indice, ponto in enumerate(np.array(pontos)):
		distancia_euclidiana = np.linalg.norm(ponto_zero - ponto)
		if distancia_euclidiana < menor_distancia:
			menor_distancia = distancia_euclidiana
			indice_menor_distancia = indice
		if distancia_euclidiana > maior_distancia:
			maior_distancia = distancia_euclidiana
			indice_maior_distancia = indice
	return pontos[indice_menor_distancia], pontos[indice_maior_distancia]



if __name__ == '__main__':

	try:
		nome_arquivo = 'lote1_1703_5_gt_clean.xml'
		caminho_arquivo = '/media/jones/dataset/alpr/lotes_rotulacao/l1/amostras_l5'
		caminho_completo_arquivo = os.path.abspath(os.path.join(caminho_arquivo, nome_arquivo))
		annotation_tree = ET.parse(caminho_completo_arquivo)
		tag_raiz = annotation_tree.getroot()
		qtd_imagens_rotuladas = 0
		qtd_imagens = 0
		# [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
		dict_bbox_gt = {}
		for indice_tag_image, image_tag in enumerate(tag_raiz.findall('image')):
			nome_frame = image_tag.get('name')
			boxes = image_tag.findall('polygon')
			lista_bbox_gt_frame = []
			for box in boxes:
				if 'plate' in box.get('label'):
					# forma de obtencao do bbox qdo o shape e retangulo
					# bbox_plate = [int(float(box.get('xtl'))), int(float(box.get('ytl'))), int(float(box.get('xbr'))), int(float(box.get('ybr'))), 0, 0, 0]
					# forma de obtencao do bbox qdo o shape e poligon 4 pontos
					pontos_str = box.get('points').split(';')
					pontos = [point_str_to_tuple(p) for p in pontos_str]
					ponto_top_left, ponto_bottom_right = get_pontos_extremos(pontos)
					bbox_plate = [ponto_top_left[0], ponto_top_left[1], ponto_bottom_right[0], ponto_bottom_right[1], 0, 0, 0]
					lista_bbox_gt_frame.append(bbox_plate)
					print(image_tag.tag, image_tag.attrib, box.get('label'))
			dict_bbox_gt[nome_frame] = np.array(lista_bbox_gt_frame)


		input_dir  = sys.argv[1]
		output_dir = input_dir

		lp_threshold = .2

		wpod_net_path = sys.argv[2]
		wpod_net = load_model(wpod_net_path)

		imgs_paths = glob('%s/*.jpg' % input_dir)

		print('Searching for license plates using WPOD-NET')

		for i,img_path in enumerate(imgs_paths):

			print('\t Processing %s' % img_path)

			bname = splitext(basename(img_path))[0]
			name_file = basename(img_path)
			Ivehicle = cv2.imread(img_path)
			height, width = Ivehicle.shape[:2]
			ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
			side  = int(ratio*288.)
			bound_dim = min(side + (side%(2**4)),608)
			print("\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio))

			Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,(240,80),lp_threshold)
			ground_truth_frame = dict_bbox_gt[name_file]
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
					writeShapes('%s/%s_lp.txt' % (output_dir,bname),[s])
				metric_fn.add(np.array(lista_preds_frame), ground_truth_frame)
			else:
				gt_frame = dict_bbox_gt[name_file]
				# metric_fn.add(np.empty(()), gt_frame)
			if is_exibir_gt and len(LlpImgs):
				for bbox_gt in ground_truth_frame:
					cv2.rectangle(car_rgb_np, (bbox_gt[0], bbox_gt[1]), (bbox_gt[2], bbox_gt[3]), bbox_color_gt, thickness)  # filled
				cv2.imwrite('%s/%s_lp_car.png' % (output_dir, bname), car_rgb_np)
		print('calculando mAP')
		# compute PASCAL VOC metric
		print('VOC PASCAL mAP:'+str({metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']}))
		# compute PASCAL VOC metric at the all points
		print('VOC PASCAL mAP in all points:'+str({metric_fn.value(iou_thresholds=0.5)['mAP']}))
		# compute metric COCO metric
		print('COCO mAP: '+str({metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']}))

	except Exception as error:
		print(error)
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)


