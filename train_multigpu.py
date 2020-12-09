import tensorflow as tf
from tensorflow import keras
import detection.config as cfg
from detection.datasets import coco, data_generator
from detection.models.detectors import faster_rcnn
from pycocotools.cocoeval import COCOeval
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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

train_dataset = coco.CocoDataSet('./COCO2017/', 'train',
                                 flip_ratio=0.5,
                                 pad_mode='fixed',
                                 mean=img_mean,
                                 std=img_std,
                                 scale=(800, 1216),
                                 debug=True)


train_generator = data_generator.DataGenerator(train_dataset)
train_tf_dataset = tf.data.Dataset.from_generator(
    train_generator, (tf.float32, tf.float32, tf.float32, tf.int32))
train_tf_dataset = train_tf_dataset.batch(cfg.batch_size).prefetch(100).shuffle(100)
num_classes = len(train_dataset.get_categories())
model = faster_rcnn.FasterRCNN(num_classes=num_classes)
optimizer = keras.optimizers.SGD(cfg.learning_rate, momentum=0.9, nesterov=True)

if cfg.finetune:
    model.load_weights('./model/faster_rcnn.h5')

for epoch in range(cfg.epochs):

    for (batch, inputs) in enumerate(train_tf_dataset):

        batch_imgs, batch_metas, batch_bboxes, batch_labels = inputs

        with tf.GradientTape() as tape:
            rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss = \
                model((batch_imgs, batch_metas, batch_bboxes, batch_labels))

            loss_value = rpn_class_loss + rpn_bbox_loss + rcnn_class_loss + rcnn_bbox_loss

        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if batch % 1 == 0:
            print('Epoch:', epoch, 'Batch:', batch, 'Loss:', loss_value.numpy(),
                  'RPN Class Loss:', rpn_class_loss.numpy(),
                  'RPN Bbox Loss:', rpn_bbox_loss.numpy(),
                  'RCNN Class Loss:', rcnn_class_loss.numpy(),
                  'RCNN Bbox Loss:', rcnn_bbox_loss.numpy())

    if epoch % cfg.checkpoint == 0 or epoch == cfg.epochs:
        model.save_weights('./model/faster_rcnn.h5')
