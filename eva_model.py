import os
import json
import tensorflow as tf
import numpy as np
from pycocotool.cocoeval import COCOeval
from detection.datasets import coco, data_generator
from detection.models.detectors import faster_rcnn
from detection import config as cfg 
import visualize
from detection.datasets.utils import get_original_image
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#  try:
#    # Currently, memory growth needs to be the same across GPUs
#    for gpu in gpus:
#      tf.config.experimental.set_memory_growth(gpu, True)
#    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#  except RuntimeError as e:
#    # Memory growth must be set before GPUs have been initialized
#    print(e)

img_mean = (123.675, 116.28, 103.53)
# img_std = (58.395, 57.12, 57.375)
img_std = (1., 1., 1.)

val_dataset = coco.CocoDataSet('./COCO2017/', 'val',
                               flip_ratio=0,
                               pad_mode='fixed',
                               mean=img_mean,
                               std=img_std,
                               scale=(800, 1344))
print(len(val_dataset))


model = faster_rcnn.FasterRCNN(
    num_classes=len(val_dataset.get_categories()))


img, img_meta, bboxes, labels = val_dataset[0]
batch_imgs = tf.Variable(np.expand_dims(img, 0))
batch_metas = tf.Variable(np.expand_dims(img_meta, 0))

_ = model((batch_imgs, batch_metas), training=False)

model.load_weights('model/faster_rcnn.h5', by_name=True)

batch_size = 1

dataset_results = []
imgIds = []
for idx in range(10):
    if idx % 10 == 0:
        print(idx)
    
    img, img_meta, boxes, label = val_dataset[idx]
    ori_img = get_original_image(img, img_meta, img_mean)

    proposals = model.simple_test_rpn(img, img_meta)
    res = model.simple_test_bboxes(img, img_meta, boxes)
    
    image_id = val_dataset.img_ids[idx]
    imgIds.append(image_id)
    # print(type(img_meta))
    # visualize.display_instances(ori_img, res['rois'], res['class_ids'],
    #                         val_dataset.get_categories(), scores=res['scores'])
    print(res)
    
    for pos in range(res['class_ids'].shape[0]):
        results = dict()
        results['score'] = float(res['scores'][pos])
        results['category_id'] = val_dataset.label2cat[int(res['class_ids'][pos])]
        y1, x1, y2, x2 = [float(num) for num in list(res['rois'][pos])]
        results['bbox'] = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
        results['image_id'] = image_id
        dataset_results.append(results)
# print(dataset_results)

if not dataset_results == []:
    with open('./result/epoch_' + str(cfg.epochs) + '.json', 'w') as f:
        f.write(json.dumps(dataset_results))

   coco_dt = val_dataset.coco.loadRes('epoch_' + str(cfg.epochs) + '.json')
   cocoEval = COCOeval(val_dataset.coco, coco_dt, 'bbox')
   cocoEval.params.imgIds = imgIds

   cocoEval.evaluate()
   cocoEval.accumulate()
   cocoEval.summarize()
   with open('evaluation.txt', 'a+') as f:
       content = 'Epoch: ' + str(cfg.epochs) + '\n' + str(cocoEval.stats) + '\n'
       f.write(content)
