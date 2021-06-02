from evaluate_model_async import calc_iou



def analisar_calc_iou():
	#gt = [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
	# gt = [[875, 204, 916, 207, 0, 0, 0]]
	gt = [[875, 185, 916, 224, 0, 0, 0]]
	gt_bbox = (gt[0][0], gt[0][1], gt[0][2], gt[0][3])
	#pred = [xmin, ymin, xmax, ymax, class_id, confidence]
	pred = [[875, 189, 918, 224, 0, 0.9999641]]
	pred_bbox = (pred[0][0], pred[0][1], pred[0][2], pred[0][3])
	iou = calc_iou(gt_bbox, pred_bbox)
	print('iou: %f.02' % iou)


analisar_calc_iou()