from numpy import expand_dims
from numpy import mean
from mrcnn.utils import compute_ap
from mrcnn.model import mold_image
from mrcnn.model import load_image_gt
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import xml.etree.ElementTree as ET
import collections
import pandas as pd
import seaborn as sns
import glob

# Calculate the mAP for a model on a given dataset
def evaluate_model(dataset, model, cfg):
    APs = list()
    for image_id in dataset.image_ids:
        # load image, bounding boxes and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)
        # extract results for first sample
        r = yhat[0]
        # calculate statistics, including AP
        AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        # store
        APs.append(AP)
    # calculate the mean AP across all images
    mAP = mean(APs)
    return mAP
    
# plot a number of photos with ground truth and predictions
def plot_actual_vs_predicted(dataset, model, cfg, n_images=5):
    # load image and mask
    for i in range(n_images):
        # load the image and mask
        image = dataset.load_image(i)
        mask, _ = dataset.load_mask(i)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)[0]
        # define subplot
        plt.figure(figsize=(20, 70))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,hspace=.01)
        plt.subplot(n_images, 2, i*2+1)
        # plot raw pixel data
        plt.imshow(image)
        plt.title('Actual')
        plt.xticks(())
        plt.yticks(())
        # plot masks
        for j in range(mask.shape[2]):
            plt.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
        # get the context for drawing boxes
        plt.subplot(n_images, 2, i*2+2)
        # plot raw pixel data
        plt.imshow(image)
        plt.title('Predicted')
        plt.xticks(())
        plt.yticks(())
        ax = plt.gca()
        # plot each box
        for box in yhat['rois']:
            # get coordinates
            y1, x1, y2, x2 = box
            # calculate width and height of the box
            width, height = x2 - x1, y2 - y1
            # create the shape
            rect = Rectangle((x1, y1), width, height, fill=False, color='red')
            # draw the box
            ax.add_patch(rect)
    # show the figure
    plt.show()
    #plt.savefig('foo.png')
    #plt.savefig('foo.pdf')



# funciton to modify xml files
def fix_path(ANNOT_DIR, IMAGE_DIR):
    for f in glob.glob(ANNOT_DIR + '*.xml'):
        tree = ET.parse(f)
        for elem in tree.iter('path'):
            elem.text = IMAGE_DIR + f.split('/')[-1].split('.')[0] + '.jpg'  # All files extensions in lowercase
        tree.write(f, xml_declaration=True, method='xml', encoding="utf8")

def fix_filename(PATH):
    for f in glob.glob(PATH + '*.xml'):
        tree = ET.parse(f)
        for elem in tree.iter('filename'):
            elem.text = f.split('/')[-1].split('.')[0] + '.jpg'
        tree.write(f, xml_declaration=True, method='xml', encoding="utf8")
        
def fix_folder(PATH, folder='images'):
    for f in glob.glob(PATH + '*.xml'):
        tree = ET.parse(f)
        for elem in tree.iter('folder'):
            elem.text = folder
        tree.write(f, xml_declaration=True, method='xml', encoding="utf8")
        
def list_labels(PATH):
    labels = []
    for f in glob.glob(PATH + '*.xml'):
        #adding the encoding when the file is opened and written is needed to avoid a charmap error
        tree = ET.parse(f)
        for elem in tree.iter('name'):
            labels.append(elem.text)
    return labels

def one_label(PATH):
    area = max_area(PATH) + 1
    label_by_area(PATH, (area,0))

    
def set_label(PATH, label='IO'):
    for f in glob.glob(PATH + '*.xml'):
        tree = ET.parse(f)
        for elem in tree.iter('name'):
            elem.text = label
        tree.write(f, xml_declaration=True, method='xml', encoding="utf8")
        
def max_area(PATH):
    largest = 0
    for f in glob.glob(PATH + '*.xml'):
        tree = ET.parse(f)
        root = tree.getroot()
        for member in root.findall('object'): 
            xmin = int(member[4][0].text)
            ymin = int(member[4][1].text)
            xmax = int(member[4][2].text)
            ymax = int(member[4][3].text)
            area = (xmax - xmin) * (ymax - ymin)
            if area > largest:
                largest = area
    return largest

def min_area(PATH):
    smallest = max_area(PATH)
    for f in glob.glob(PATH + '*.xml'):
        tree = ET.parse(f)
        root = tree.getroot()
        for member in root.findall('object'): 
            xmin = int(member[4][0].text)
            ymin = int(member[4][1].text)
            xmax = int(member[4][2].text)
            ymax = int(member[4][3].text)
            area = (xmax - xmin) * (ymax - ymin)
            if area < smallest:
                smallest = area
    return smallest

def label_by_area(PATH, bounds=(125,250)):
    S , M  = bounds
    for f in glob.glob(PATH + '*.xml'):
        tree = ET.parse(f)
        root = tree.getroot()
        for member in root.findall('object'): 
            root.find('filename').text
            int(root.find('size')[0].text)
            int(root.find('size')[1].text)
            xmin = int(member[4][0].text)
            ymin = int(member[4][1].text)
            xmax = int(member[4][2].text)
            ymax = int(member[4][3].text)
            area = (xmax - xmin) * (ymax - ymin)
            if area < S:
                member[0].text = 'S'
            elif area < M:
                member[0].text = 'M'
            else:
                member[0].text = 'L'
        tree.write(f, xml_declaration=True, method='xml', encoding="utf8")

# fix bboxes if image was flipped on the vertical 
def flop_boxes(PATH):
    for f in glob.glob(PATH + '*.xml'):
        tree = ET.parse(f)
        root = tree.getroot()
        for member in root.findall('object'): 
            width = int(root.find('size')[0].text)
            height = int(root.find('size')[1].text)
            xmin = int(member[4][0].text)
            ymin = int(member[4][1].text)
            xmax = int(member[4][2].text)
            ymax = int(member[4][3].text)
            box_w = xmax -xmin
            new_xmin = width - xmin - box_w
            new_xmax = width - xmax + box_w
            member[4][0].text = str(new_xmin)
            member[4][2].text = str(new_xmax)
        tree.write(f, xml_declaration=True, method='xml', encoding="utf8")

        
# load annotations to a dataframe so can perform some analysis 
def toDataFrame(PATH):
    objects = []
    for f in glob.glob(PATH + '*.xml'):
        tree = ET.parse(f)
        root = tree.getroot()
        for member in root.findall('object'): 
            filename = root.find('filename').text
            width = int(root.find('size')[0].text)
            height = int(root.find('size')[1].text)
            label = member[0].text
            xmin = int(member[4][0].text)
            ymin = int(member[4][1].text)
            xmax = int(member[4][2].text)
            ymax = int(member[4][3].text)
            cx = (xmax + xmin) / 2
            cy = (ymax + ymin) / 2
            area = (xmax - xmin) * (ymax - ymin)
            objects.append((filename, width, height, xmin, ymin, xmax, ymax, cx, cy, area, label))
    column_name = ['filename', 'width', 'height', 'xmin', 'ymin', 'xmax', 'ymax', 'cx', 'cy', 'area', 'label']
    df = pd.DataFrame(objects, columns=column_name)
    return df
