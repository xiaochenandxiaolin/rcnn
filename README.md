# Faster RCNN
Faster R-CNN Resnet-101-FPN implementation based on TensorFlow 2.3.

# Experiment
For now, I refactor code.  
That means functions in detection/utils/misc.py do work.

# Usage
- see `train_multigpu.py`, `inspect_model.py` and `eval_model.py`

# Training R-CNN models and eval

**Command Line:**
```bash
#now to run training and eval code.
python train_multigpu.py
python eval_model.py -language_eval 0
```
```bash
#Change the arguments
To train a new model,simply run:
python train.py -batch  -epochs 
```
# Updating
- [ ] grasp detection
- [ ] coco loadRes

# Acknowledgement
This work builds on many excellent works, which include:
- Heavily based on [tf-eager-fasterrcnn](https://github.com/Viredery/tf-eager-fasterrcnn)
- [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)
- [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)