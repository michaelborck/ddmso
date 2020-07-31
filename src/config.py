from mrcnn.config import Config
PROJECT_DIR = '/home/michael/Projects/ddmso/'

class Config(Config):
    DATA_PATH = PROJECT_DIR + 'data/processed/visible/rgb/Twilight'
    ANNOTS_PATH = PROJECT_DIR + 'data/processed/visible/annots'
    INITIAL_WEIGHTS = PROJECT_DIR + 'weights/mask_rcnn_coco.h5'
    MODEL_WEIGHTS =  PROJECT_DIR + 'weights/mask_rcnn_twilight_aug_cfg_0200.h5'  # Update after training
    MODEL_DIR = PROJECT_DIR + 'experiements/'
    NAME = "test_cfg"       # Give the configuration a recognizable name
    NUM_CLASSES = 1 + 1     # Number of classes (background + object)
    NUM_EPOCHS = 200
    STEPS_PER_EPOCH = 200   # Number of training steps per epoch
    # GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.01
    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.01
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7
