import os
import gc
import torch
import torch.nn.functional as F
import mmcv
from tqdm import tqdm
from mmcv.utils import print_log
from mmdet.core.visualization.image import imshow_det_bboxes
from mmseg.core import intersect_and_union, pre_eval_to_metrics
from collections import OrderedDict
from prettytable import PrettyTable
import numpy as np
import pycocotools.mask as maskUtils
from mmcv import imresize

import shutil
# TODO
from configs.mapillary_label2id_v5 import CONFIG as config_mapillary_label2id
from configs.mapillary_id2label import CONFIG as config_mapillary_id2label
from configs.mapillary_openset_v5 import CONFIG as config_mapillary_openset
from configs.cityscapes_to_mapillary import CONFIG as cs_to_mapi
from configs.mapillary_label2context_v5 import CONFIG as config_mapillary_context

from clip_utils import patch_classification, sclip_get_global_features

PALETTE = np.array([[165, 42, 42], [0, 192, 0], [196, 196, 196], [190, 153, 153],
                    [180, 165, 180], [90, 120, 150], [102, 102, 156],
                    [128, 64, 255], [140, 140, 200], [170, 170, 170],
                    [250, 170, 160], [96, 96, 96],
                    [230, 150, 140], [128, 64, 128], [110, 110, 110],
                    [244, 35, 232], [150, 100, 100], [70, 70, 70], [150, 120, 90],
                    [220, 20, 60], [255, 0, 0], [255, 0, 100], [255, 0, 200],
                    [200, 128, 128], [255, 255, 255], [64, 170,
                                                       64], [230, 160, 50],
                    [70, 130, 180], [190, 255, 255], [152, 251, 152],
                    [107, 142, 35], [0, 170, 30], [255, 255, 128], [250, 0, 30],
                    [100, 140, 180], [220, 220, 220], [220, 128, 128],
                    [222, 40, 40], [100, 170, 30], [40, 40, 40], [33, 33, 33],
                    [100, 128, 160], [142, 0, 0], [70, 100, 150], [210, 170, 100],
                    [153, 153, 153], [128, 128, 128], [0, 0, 80], [250, 170, 30],
                    [192, 192, 192], [220, 220, 0], [140, 140, 20], [119, 11, 32],
                    [150, 0, 255], [0, 60, 100], [0, 0, 142], [0, 0, 90],
                    [0, 0, 230], [0, 80, 100], [128, 64, 64], [0, 0, 110],
                    [0, 0, 70], [0, 0, 192], [32, 32, 32], [120, 10,
                                                            10], [0, 0, 0]])

# TO redefine here
LARGE_SCALE_CLASSES = ["road"]
MEDIUM_SCALE_CLASSES = ["sidewalk", "vegetation", "terrain", "sky"]
SMALL_SCALE_CLASSES = ["building",
                       "fence", "pole"]

mapi_large_scale_class = ["Road", ]
mapi_medium_scale_class = ["Sidewalk", "Bike Lane", "Curb", "Pedestrian Area", "Rail Track",
                           "Bridge", "Road Bridge",
                           "Mountain", "Sand", "Sky", "Snow", "Terrain", "Vegetation", "Water", ]
mapi_small_scale_class = ["Wall", "Building",
                          "Traffic Sign Frame",
                          "Lane Marking - Crosswalk", "Lane Marking - General",
                          "Fence", "Guard Rail",
                          "Phonebooth", "Pole"
                          ]


def load_filename_with_extensions(data_path, filename):
    """
    Returns file with corresponding extension to json file.
    Raise error if such file is not found.

    Args:
        filename (str): Filename (without extension).

    Returns:
        filename with the right extension.
    """
    full_file_path = os.path.join(data_path, filename)
    # List of image file extensions to attempt
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
    # Iterate through image file extensions and attempt to upload the file
    for ext in image_extensions:
        # Check if the file with current extension exists
        if os.path.exists(full_file_path + ext):
            return full_file_path + ext  # Return True if file is successfully uploaded
    raise FileNotFoundError(f"No such file {full_file_path}, checked for the following extensions {image_extensions}")


def img_load(data_path, filename, dataset):
    # load image
    if dataset == 'ade20k':
        img = mmcv.imread(os.path.join(data_path, filename + '.jpg'))
    elif dataset == 'cityscapes' or dataset == 'foggy_driving':
        img = mmcv.imread(os.path.join(data_path, filename + '.png'))
    else:
        raise NotImplementedError()
    return img


def eval_pipeline(gt_path, res_path, dataset):
    logger = None
    if dataset == 'cityscapes' or dataset == 'foggy_driving':
        class_names = (
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation',
            'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')
    file_client = mmcv.FileClient(**{'backend': 'disk'})
    pre_eval_results = []
    if dataset == 'cityscapes':
        # TODO
        prefixs = ['frankfurt', 'lindau', 'munster']
        # prefixs = ['lindau']
    elif dataset == 'foggy_driving':
        prefixs = ['public', 'pedestrian']
    else:
        raise NotImplementedError
    for split in tqdm(prefixs, desc="Split loop"):
        gt_path_split = os.path.join(gt_path, split)
        res_path_split = os.path.join(res_path, split)
        filenames = [fn_ for fn_ in os.listdir(res_path_split) if '.json' in fn_]
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
            if dataset == 'cityscapes' or dataset == 'foggy_driving':
                gt_fn_ = os.path.join(gt_path_split, fn_.replace('_leftImg8bit_semantic.json', '_gtFine_labelTrainIds.png'))
            elif dataset == 'ade20k':
                gt_fn_ = os.path.join(gt_path, fn_.replace('_semantic.json', '.png'))
            img_bytes = file_client.get(gt_fn_)
            seg_map = mmcv.imfrombytes(
                img_bytes, flag='unchanged',
                backend='pillow').squeeze().astype(np.uint8)
            if dataset == 'ade20k':
                seg_map = seg_map - 1
            pre_eval_results.append(intersect_and_union(
                seg_pred,
                seg_map,
                num_classes,
                255,
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


def convert_index(label_img, label_map):
    label_img_new = torch.zeros_like(label_img)
    labels = torch.unique(label_img)
    # print("original labels: ", labels)
    for label in labels:
        label_new = label_map[label.item()]
        label_mask = label_img == label
        # print(label_mask.shape, np.unique(label_mask))
        label_img_new[label_mask] = label_new
    # labels_new = np.unique(label_img_new)
    # print("new labels: ", labels_new)
    return label_img_new


def class_scale(top_1_id):
    # TODO add multiple scales
    # original_class = config_mapillary_id2label["id2label"][str(top_1_id)]
    original_class = top_1_id
    if original_class in LARGE_SCALE_CLASSES:
        # print("==========", top_1_propose_class_names[0])
        patch_padding = 120
    elif original_class in MEDIUM_SCALE_CLASSES:
        patch_padding = 80
    elif original_class in SMALL_SCALE_CLASSES:
        patch_padding = 40
    else:
        patch_padding = 10
    # patch_padding = patch_padding + 60
    return patch_padding


def semantic_segment_anything_inference_clip(output_path, data=None, save_img=False,
                                             semantic_branch_model=None,
                                             mask_branch_model=None,
                                             id2label=None,
                                             clip_model=None,
                                             prob_threshold=0.0,):
    torch.cuda.empty_cache()
    # 1. read files
    file_name = data['img_metas'][0].data[0][0]["filename"]
    filename = data['img_metas'][0].data[0][0]["ori_filename"][:-4]
    print("************READ FROM", file_name)
    # print(data['img_metas'][0].data[0][0]["ori_filename"])
    # print(filename)
    img_original = mmcv.imread(file_name)
    img = imresize(img_original, [1920, 1440])

    # 2. get anntations from SAM and segmentation masks
    anns = {'annotations': mask_branch_model.generate(img)}
    h, w, _ = img.shape
    class_names = []
    with torch.no_grad():
        results = semantic_branch_model(return_loss=False, **data)
        class_ids_original = torch.from_numpy(results[0])
        del results
        class_ids = torch.nn.functional.interpolate(class_ids_original[None, None, :].float(), size=[1440, 1920],
                                                    mode='nearest', align_corners=None, ).long().squeeze().cuda()

    # TODO first convert all the labels to mapi one
    class_ids_new = convert_index(class_ids, cs_to_mapi["cs_to_mapi"])
    semantc_mask = class_ids_new.clone()

    anns['annotations'] = sorted(anns['annotations'], key=lambda x: x['area'], reverse=True)

    for i, ann in enumerate(anns['annotations']):
        valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
        propose_classes_ids = class_ids[valid_mask]
        top_1_propose_class_ids = torch.bincount(propose_classes_ids.flatten()).topk(1).indices  # cityscapes id
        top_1_propose_class_names = [id2label['id2label'][str(class_id.item())] for class_id in
                                     top_1_propose_class_ids]  # cityscapes id2label
        top_1_mapi_id = cs_to_mapi['cs_to_mapi'][top_1_propose_class_ids[0].item()]  # cityscapes name to mapillary name
        patch_padding = class_scale(top_1_propose_class_names[0])

        # 2. get open-set mapillary class names from the coarse label
        op_class_list = list(config_mapillary_openset["mapillary_openset"][top_1_propose_class_names[0]])
        unlabeled = list(config_mapillary_openset["mapillary_openset"]["unlabeled"])
        local_class_list = op_class_list
        local_class_list.extend(unlabeled)

        # 3. TODO add context renovation names for clip_classification
        name_renovation = True
        if True:
            local_class_list_extended = []
            for tmp_name in local_class_list:
                tmp_name_extended = config_mapillary_context["mapillary_context"][tmp_name]
                local_class_list_extended.append(list(tmp_name_extended))

        img_processed = img
        feature_G = sclip_get_global_features(img_processed, ann, w, h, patch_padding, clip_model, random_padding=True)

        large_categories, large_probs = patch_classification(top_k=0,
                                                             patch_padding=0,
                                                             image=img_processed,
                                                             masked_img=valid_mask,
                                                             ann=ann,
                                                             class_list=local_class_list,
                                                             class_list_extended=local_class_list_extended,
                                                             w=w, h=h, clip_model=clip_model,
                                                             name_renovation=name_renovation,
                                                             global_feature=feature_G,
                                                             )
        probs = large_probs

        del feature_G

        # 6. update if the prob is above a certain threshold
        if torch.max(probs) > prob_threshold:
            index = torch.argmax(probs)
            top1_category = large_categories[index]
        else:
            top1_category = config_mapillary_id2label["id2label"][str(top_1_mapi_id)]

        # 7. convert the ids and class names
        top1_id = int(config_mapillary_label2id['label2id'][top1_category])
        top_1_propose_class_names[0] = config_mapillary_id2label['id2label'][str(top1_id)]
        top_1_propose_class_ids = top1_id
        semantc_mask[valid_mask] = top_1_propose_class_ids
        ann['class_name'] = top_1_propose_class_names[0]
        ann['class_proposals'] = top_1_propose_class_names[0]
        class_names.append(ann['class_name'])

        del valid_mask
        del propose_classes_ids
        del top_1_propose_class_ids
        del top_1_propose_class_names
        torch.cuda.empty_cache()

    # 8. put everything back in original resolution
    img = img_original
    h_ori, w_ori, _ = img.shape
    semantc_mask = torch.nn.functional.interpolate(semantc_mask[None, None, :].float(), size=[h_ori, w_ori],
                                                   mode='nearest', align_corners=None, ).long().squeeze()
    class_ids = class_ids_original
    # print("************ORIGINAL", img.shape, semantc_mask.shape, class_ids.shape)
    sematic_class_in_img = torch.unique(semantc_mask)
    semantic_bitmasks, semantic_class_names = [], []
    anns['semantic_mask'] = {}
    for i in range(len(sematic_class_in_img)):
        # change to mapi labels
        class_name = config_mapillary_id2label['id2label'][str(sematic_class_in_img[i].item())]
        class_mask = semantc_mask == sematic_class_in_img[i]
        class_mask = class_mask.cpu().numpy().astype(np.uint8)
        semantic_class_names.append(class_name)
        semantic_bitmasks.append(class_mask)
        anns['semantic_mask'][str(sematic_class_in_img[i].item())] = maskUtils.encode(
            np.array((semantc_mask == sematic_class_in_img[i]).cpu().numpy(), order='F', dtype=np.uint8))
        anns['semantic_mask'][str(sematic_class_in_img[i].item())]['counts'] = \
            anns['semantic_mask'][str(sematic_class_in_img[i].item())]['counts'].decode('utf-8')

    # 9. save predictions
    if save_img:
        imshow_det_bboxes(img,
                          bboxes=None,
                          labels=np.arange(len(sematic_class_in_img)),
                          segms=np.stack(semantic_bitmasks),
                          class_names=semantic_class_names,
                          font_size=25,
                          show=False,
                          out_file=os.path.join(output_path, 'visualization', filename + '.png'))
        print('[Save] save SSA prediction: ', os.path.join(output_path, 'visualization', filename + '.png'))
    save_original_seg = False
    # original_classes = torch.unique(class_ids)
    if save_original_seg:
        original_classes = torch.unique(class_ids)
        original_bitmasks, original_class_names = [], []
        for i in range(len(original_classes)):
            # change to original labels
            class_name = id2label['id2label'][str(original_classes[i].item())]
            class_mask = class_ids == original_classes[i]
            class_mask = class_mask.cpu().numpy().astype(np.uint8)
            original_class_names.append(class_name)
            original_bitmasks.append(class_mask)

        imshow_det_bboxes(img,
                          bboxes=None,
                          labels=np.arange(len(original_classes)),
                          segms=np.stack(original_bitmasks),
                          class_names=original_class_names,
                          font_size=25,
                          show=False,
                          out_file=os.path.join(output_path, filename + '_semantic_original.png'))
        shutil.copy(file_name, os.path.join(output_path, filename + '_original.png'))
        print('[Save] save SSA prediction: ',
              os.path.join(output_path, '/original_seg/', filename + '_semantic_original.png'))
        del original_classes
        del original_bitmasks
        del original_class_names

    shutil.copy(file_name, os.path.join(output_path, 'images', filename + '.png'))
    semantc_mask = semantc_mask.detach().cpu().numpy()
    mmcv.image.imwrite(semantc_mask.astype(np.uint8), os.path.join(output_path, 'labels', filename + '.png'))

    mmcv.dump(anns, os.path.join(output_path, 'json', filename + '_semantic.json'))

    # 手动清理不再需要的变量
    del img
    del data
    del anns
    del class_ids
    del semantc_mask
    del class_names
    del semantic_bitmasks
    del semantic_class_names

    gc.collect()
