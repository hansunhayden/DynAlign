import os
import glob
from PIL import Image
import numpy as np
import cv2

PALETTE = [[165, 42, 42], [0, 192, 0], [196, 196, 196],
           [190, 153, 153], [180, 165, 180], [90, 120, 150],
           [102, 102, 156], [128, 64, 255], [140, 140, 200],
           [170, 170, 170], [250, 170, 160], [96, 96, 96],
           [230, 150, 140], [128, 64, 128], [110, 110, 110],
           [244, 35, 232], [150, 100, 100], [70, 70, 70],
           [150, 120, 90], [220, 20, 60], [255, 0, 0],
           [255, 0, 100], [255, 0, 200], [200, 128, 128],
           [255, 255, 255], [64, 170, 64], [0, 0, 0]]

# CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
#            'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
#            'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
#            'bicycle',
#            'unlabeled')
CLASSES = ('road', 'drivable fallback', 'sidewalk',
            'non-drivable fallback', 'person', 'rider',
            'motorcycle', 'bicycle', 'autorickshaw',
            'car', 'truck', 'bus',
           'vehicle fallback', 'curb', 'wall',
           'fence', 'guard rail', 'billboard',
           'traffic sign', 'traffic light', 'pole',
           'obs-str-bar-fallback', 'building', 'bridge',
           'vegetation', 'sky', 'unlabeled')

# map_label_path = "/scratch/izar/hansun/Code_Data/segmentation/Mapillary_1"
# map_label_path_new = "/scratch/izar/hansun/Code_Data/segmentation/Mapillary_1_new"
label_path = "/home/hansun/nas/hansun/DATA/segmentation/IDD_Segmentation/gtFine/val"
vis_path = "/home/hansun/nas/hansun/DATA/segmentation/IDD_Segmentation/gtFine/val_vis"


def apply_color_map(label_img, palette):
    color_array = np.zeros((label_img.shape[0], label_img.shape[1], 3), dtype=np.uint8)
    for i in range(len(palette)):
        # set all pixels with the current label to the color of the current label
        color_array[label_img == i] = palette[i]
    return color_array

if __name__ == "__main__":
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)
    for dir in [x[0] for x in os.walk(label_path)]:
        current_path = os.path.join(label_path, dir)
        print(current_vis_path)
        current_vis_path = os.path.join(vis_path, dir)
        if not os.path.exists(current_vis_path):
            os.makedirs(current_vis_path)
        images = glob.glob(current_path + '/*_gtFine_labellevel3Ids.png')
        for image in images:
            label_img = np.array(Image.open(image).convert('L'))

            vis_img_path = os.path.join(current_vis_path, os.path.basename(image))
            vis_label = apply_color_map(label_img, PALETTE)
            vis_im = Image.fromarray(vis_label)
            vis_im.save(vis_img_path)

            print(vis_img_path, np.unique(label_img))
        # break
    print("*******conversion FINISHED")
