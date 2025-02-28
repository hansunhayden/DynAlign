# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from mmseg.core import add_prefix
from mmseg.ops import resize
from mmcv.utils import print_log

import os
import mmcv
import argparse
import numpy as np
from collections import OrderedDict
import pycocotools.mask as maskUtils
from prettytable import PrettyTable
from torchvision.utils import save_image, make_grid
from mmseg.core import eval_metrics, pre_eval_to_metrics

def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray | str): Prediction segmentation map
            or predict result filename.
        label (ndarray | str): Ground truth segmentation map
            or label filename.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. The parameter will
            work only when label is str. Default: False.

     Returns:
         torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
         torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
         torch.Tensor: The prediction histogram on all classes.
         torch.Tensor: The ground truth histogram on all classes.
    """

    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    if isinstance(label, str):
        label = torch.from_numpy(
            mmcv.imread(label, flag='unchanged', backend='pillow'))
    else:
        label = torch.from_numpy(label)

    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    # if isinstance(ignore_index, list):
    #     # print("==========================", ignore_index)
    #     mask = (label != ignore_index[0])
    #     for ig_label in ignore_index:
    #         mask = mask * (label != ig_label)
    # else:
    #     # print("==========================", ignore_index)
    #     mask = (label != ignore_index)
    if isinstance(ignore_index, list):
        # print(ignore_index)
        for idx in ignore_index:
            # print(idx)
            label[label == idx] = 255
    # print(np.unique(label))
    mask = (label != 255)
    pred_label = pred_label[mask]
    label = label[mask]
    print(np.unique(label))

    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect

    # print("=============", area_intersect)
    return area_intersect, area_union, area_pred_label, area_label

def parse_args():
    parser = argparse.ArgumentParser(description='Semantically segment anything.')
    parser.add_argument('--gt_path', help='the directory of gt annotations')
    parser.add_argument('--result_path', help='the directory of semantic predictions')
    parser.add_argument('--dataset', type=str, default='cityscapes',
                        choices=['cityscapes', 'foggy_driving', 'mapillary-closed', 'mapillary-open', "idd", "idd-train"],
                        help='specify the dataset')
    parser.add_argument('--gta', default=False, action='store_true')
    args = parser.parse_args()
    return args


args = parse_args()
logger = None
if args.dataset == 'cityscapes' or args.dataset == 'foggy_driving':
    class_names = (
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation',
        'terrain',
        'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')
elif args.dataset == 'mapillary-closed':
    class_names = (
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation',
        'terrain',
        'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'unlabeled')
elif args.dataset == "mapillary-open":
    class_names = (
                   'Bird', 'Ground Animal',
                   'Curb', 'Fence',
                   'Guard Rail','Barrier',
                   'Wall', 'Bike Lane',
                   'Crosswalk - Plain',
                   'Curb Cut', 'Parking',
                   'Pedestrian Area', 'Rail Track',
                   'Road',
                   'Service Lane',
                   'Sidewalk', 'Bridge', 'Building',
                   'Tunnel', 'Person', 'Bicyclist', 'Motorcyclist',
                   'Other Rider',
                   'Lane Marking - Crosswalk',
                   'Lane Marking - General', 'Mountain', 'Sand', 'Sky', 'Snow',
                   'Terrain', 'Vegetation', 'Water', 'Banner', 'Bench',
                   'Bike Rack',
                   'Billboard',
                   'Catch Basin', 'CCTV Camera','Fire Hydrant', 'Junction Box', 'Mailbox',
                   'Manhole',
                   'Phone Booth', 'Pothole', 'Street Light', 'Pole',
                   'Traffic Sign Frame', 'Utility Pole', 'Traffic Light',
                   'Traffic Sign (Back)', 'Traffic Sign (Front)', 'Trash Can',
                   'Bicycle', 'Boat', 'Bus', 'Car', 'Caravan', 'Motorcycle',
                   'On Rails',
                   'Other Vehicle',
                   'Trailer', 'Truck',
                   'Wheeled Slow', 'Car Mount', 'Ego Vehicle', 'Unlabeled'
    )
elif args.dataset == "idd" or args.dataset == "idd-train":
    class_names = ('road', 'drivable fallback', 'sidewalk', 'non-drivable fallback', 'person', 'rider',
                   'motorcycle', 'bicycle', 'autorickshaw', 'car', 'truck', 'bus', 'vehicle fallback', 'curb',
                   'wall', 'fence', 'guard rail', 'billboard', 'traffic sign', 'traffic light', 'pole',
                   'obs-str-bar-fallback', 'building', 'bridge', 'vegetation', 'sky', 'unlabeled'
                   )

file_client = mmcv.FileClient(**{'backend': 'disk'})
pre_eval_results = []
gt_path = args.gt_path
res_path = args.result_path
if args.dataset == 'cityscapes':
    # TODO
    if args.gta:
        prefixs = ['']
    else:
        prefixs = ['frankfurt', 'lindau', 'munster']
elif args.dataset == 'foggy_driving':
    prefixs = ['public', 'pedestrian']
elif args.dataset == 'mapillary-closed':
    prefixs = ['']
elif args.dataset == 'mapillary-open':
    prefixs = ['']
elif args.dataset == "idd":
    prefixs = ['119', '132', '147', '148', '149', '150', '153', '167', '17', '172', '18', '181', '21', '24', '3', '47',
               '51', '62', '66', '67', '88', '89']
elif args.dataset == "idd-train":
    prefixs = ["0", "1", "10", "100", "101", "102", "104", "106", "107", "108",
               "11", "110", "115", "116", "117", "118",
               "121", "122", "124", "125", "126", "127", "128", "130",
               "131", "133", "135", "138", "139",
               "140", "141", "142"]
else:
    raise NotImplementedError

ignore = 255
for split in tqdm(prefixs, desc="Split loop"):
    gt_path_split = os.path.join(gt_path, split)
    res_path_split = os.path.join(res_path, split)
    if not os.path.exists(res_path_split):
        print(res_path_split, "NOT GENERATED")
        continue
    filenames = [fn_ for fn_ in os.listdir(res_path_split) if '.json' in fn_]
    # print(filenames)
    for i, fn_ in enumerate(tqdm(filenames, desc="File loop")):
        pred_fn = os.path.join(res_path_split, fn_)
        result = mmcv.load(pred_fn)
        num_classes = len(class_names)
        init_flag = True
        for id_str, mask in result['semantic_mask'].items():
            mask_ = maskUtils.decode(mask)
            h, w = mask_.shape
            if init_flag:
                seg_mask = torch.zeros((1, 1, h, w))
                init_flag = False
            mask_ = torch.from_numpy(mask_).unsqueeze(0).unsqueeze(0)
            seg_mask[mask_] = int(id_str)
        seg_logit = torch.zeros((1, num_classes, h, w))
        seg_logit.scatter_(1, seg_mask.long(), 1)
        seg_logit = seg_logit.float()
        seg_pred = F.softmax(seg_logit, dim=1).argmax(dim=1).squeeze(0).numpy()
        if args.dataset == 'cityscapes' or args.dataset == 'foggy_driving':
            # TODO
            if args.gta:
                gt_fn_ = os.path.join(gt_path, fn_.replace('_semantic.json', '_labelTrainIds.png'))
            else:
                gt_fn_ = os.path.join(gt_path_split,
                                      fn_.replace('_leftImg8bit_semantic.json', '_gtFine_labelTrainIds.png'))
            # gt_fn_ = os.path.join(gt_path_split, fn_.replace('_leftImg8bit_semantic.json', '_gtFine_labelIds.png'))
        elif args.dataset == "mapillary-closed" or args.dataset == "mapillary-open":
            # print("================mapi", fn_)
            # TODO
            gt_fn_ = os.path.join(gt_path, fn_.replace('_semantic.json', '.png'))
            if not os.path.exists(gt_fn_):
                gt_fn_ = os.path.join(gt_path, fn_.replace('_semantic.json', 'g.png'))
            # print(gt_fn_)
            # ignore = [0, 1, 5, 7, 8, 9, 10, 11, 14, 22, 34, 37, 38, 39, 40, 59, 62, 63, 64, 65]
            ignore = [0, 1, 5, 7, 8, 9, 10, 11, 14, 22, 34, 37, 38, 39, 40, 59, 62, 63, 64, 65]
        elif args.dataset == 'ade20k':
            gt_fn_ = os.path.join(gt_path, fn_.replace('_semantic.json', '.png'))
        elif args.dataset == "idd" or args.dataset == "idd-train":
            gt_fn_ = os.path.join(gt_path_split, fn_.replace('_leftImg8bit_semantic.json', '_gtFine_labellevel3Ids.png'))
        # TODO
        if not os.path.exists(gt_fn_):
            print(gt_fn_, "NOT FOUND IN GT")
            continue
        img_bytes = file_client.get(gt_fn_)
        seg_map = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend='pillow').squeeze().astype(np.uint8)
        if args.dataset == 'ade20k':
            seg_map = seg_map - 1
        pre_eval_results.append(intersect_and_union(
            seg_pred,
            seg_map,
            num_classes,
            # 255,
            ignore,
            label_map=dict(),
            reduce_zero_label=False))

ret_metrics = pre_eval_to_metrics(pre_eval_results, ['mIoU'])
ret_metrics_summary = OrderedDict({
    ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
    for ret_metric, ret_metric_value in ret_metrics.items()
})
# each class table
ret_metrics.pop('aAcc', None)
ret_metrics_class = OrderedDict({
    ret_metric: np.round(ret_metric_value * 100, 2)
    for ret_metric, ret_metric_value in ret_metrics.items()
})
ret_metrics_class.update({'Class': class_names})
ret_metrics_class.move_to_end('Class', last=False)

for key, val in ret_metrics_class.items():
    print(key, ": ", val)
# for logger
class_table_data = PrettyTable()
for key, val in ret_metrics_class.items():
    class_table_data.add_column(key, val)

summary_table_data = PrettyTable()
for key, val in ret_metrics_summary.items():
    if key == 'aAcc':
        summary_table_data.add_column(key, [val])
    else:
        summary_table_data.add_column('m' + key, [val])

print_log('per class results:', logger)
print_log('\n' + class_table_data.get_string(), logger=logger)
print_log('Summary:', logger)
print_log('\n' + summary_table_data.get_string(), logger=logger)
