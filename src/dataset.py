from sklearn.model_selection import train_test_split
from xml.etree import ElementTree
from mrcnn.utils import Dataset
from numpy import asarray
from numpy import zeros
#from os import listdir
from os.path import basename
from glob import glob

# class that defines and loads the dataset
class Dataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, datasets, annotations, is_train=True):
        # define one class
        self.add_class("dataset", 1, 'S')
        # find all images
        images = [f for f in glob(datasets + "**/*.jpg", recursive=True)]
        x_train ,x_test = train_test_split(images,test_size=0.1, random_state=42)
        data = x_train if is_train else x_test
        for filename in data:
            # extract image id
            image_id = basename(filename)[:3]
            ann_path = annotations + image_id + '.xml'
            # add to dataset
            self.add_image('dataset', image_id=image_id, path=filename, annotation=ann_path)

    # extract bounding boxes from an annotation file
    def extract_boxes(self, filename):
        # load and parse the file
        tree = ElementTree.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        # load XML
        boxes, w, h = self.extract_boxes(path)
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('S'))
        return masks, asarray(class_ids, dtype='int32')

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']
