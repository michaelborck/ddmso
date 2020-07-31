import dataset as ds
import config as cfg
from mrcnn.model import MaskRCNN
from utils import evaluate_model

# prepare config
config = cfg.Config()
# make some local changes
config.DATA_PATH = '/home/michael/Projects/ddmso/data/processed/visible/rgb/Twilight'
config.ANNOTS_PATH = '/home/michael/Projects/ddmso/data/processed/visible/annots/'
config.MODEL_WEIGHTS = '/home/michael/Projects/ddmso/weights/mask_rcnn_twilight_aug_noflip_cfg_0200.h5'
config.display()

# train set
train_set = ds.Dataset()
train_set.load_dataset(config.DATA_PATH, config.ANNOTS_PATH, is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

# test/val set
test_set = ds.Dataset()
test_set.load_dataset(config.DATA_PATH, config.ANNOTS_PATH, is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))


# define the model
model = MaskRCNN(mode='inference', model_dir=config.MODEL_DIR, config=config)

# load weights (mscoco) and exclude the output layers
model.load_weights(config.MODEL_WEIGHTS, by_name=True)

# evaluate model on training dataset
train_mAP = evaluate_model(train_set, model, config)
print("Train mAP: %.3f" % train_mAP)

# evaluate model on test dataset
test_mAP = evaluate_model(test_set, model, config)
print("Test mAP: %.3f" % test_mAP)
