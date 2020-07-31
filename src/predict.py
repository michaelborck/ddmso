import dataset as ds
import config as cfg
from utils import plot_actual_vs_predicted
from mrcnn.model import MaskRCNN

# prepare config
config = cfg.Config()
# config.display()


# train set
train_set = ds.Dataset()
train_set.load_dataset(config.DATA_PATH, is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
# test/val set
test_set = ds.Dataset()
test_set.load_dataset(config.DATA_PATH, is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))


# define the model
model = MaskRCNN(mode='inference', model_dir=config.MODEL_DIR, config=config)
# load weights (mscoco) and exclude the output layers
model.load_weights(config.MODEL_WEIGHTS, by_name=True)

# plot predictions for train dataset
plot_actual_vs_predicted(train_set, model, config)
# plot predictions for test dataset
plot_actual_vs_predicted(test_set, model, config)
