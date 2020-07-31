from mrcnn.model import MaskRCNN
import dataset as ds
import config as cfg

# prepare config
config = cfg.Config()
# show configuration
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
model = MaskRCNN(mode='training', model_dir=config.MODEL_DIR, config=config)

# load weights (mscoco) and exclude the output layers
model.load_weights(config.INITIAL_WEIGHTS, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

# train weights (output layers or 'heads')
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=config.NUM_EPOCHS, layers='heads')
