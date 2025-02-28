import os
import argparse
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch.multiprocessing as mp
from configs.cityscapes_id2label import CONFIG as CONFIG_CITYSCAPES_ID2LABEL
from dynalign_mapillary_pipeline import semantic_segment_anything_inference_clip, eval_pipeline
from configs.mapillary_id2label import CONFIG as config_mapillary_id2label
import open_clip
device = "cuda" if torch.cuda.is_available() else "cpu"
from mmcv.utils import DictAction
from seg.HRDA.tools.test import seg


os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12322'

def parse_args():
    parser = argparse.ArgumentParser(description='Semantically segment anything.')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument(
        '--inference-mode',
        choices=[
            'same',
            'whole',
            'slide',
        ],
        default='same',
        help='Inference mode.')
    parser.add_argument('--dataset', default='Config')
    parser.add_argument(
        '--model',
        choices=[
            'model',
            'ema_model',
        ],
        default='model',
        help='Submodel to evaluate.')
    parser.add_argument(
        '--train-set',
        action='store_true',
        help='Run inference on the train set')
    parser.add_argument(
        '--test-set',
        action='store_true',
        help='Run inference on the test set')
    parser.add_argument(
        '--hrda-out',
        choices=['', 'LR', 'HR', 'ATT'],
        default='',
        help='Extract LR and HR predictions from HRDA architecture.')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
             'useful when you want to format the result to a specific format and '
             'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
             ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
             'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--data_dir', help='specify the root path of images and masks')
    parser.add_argument('--sam_path', default='ckp/sam_vit_h_4b8939.pth',
                        help='specify the root path of SAM checkpoint')
    parser.add_argument('--out_dir', help='the dir to save semantic annotations')
    parser.add_argument('--save_img', default=False, action='store_true', help='whether to save annotated images')
    parser.add_argument('--world_size', type=int, default=0, help='number of nodes')
    parser.add_argument('--dataset_sam', type=str, default='cityscapes',
                        choices=['cityscapes', 'foggy_driving', "mapillary_openset", "mapillary_closeset"],
                        help='specify the set of class names')
    parser.add_argument('--eval_sam', default=False, action='store_true', help='whether to execute evalution')
    parser.add_argument('--gt_path', default=None, help='specify the path to gt annotations')
    # parser.add_argument('--model', type=str, default='segformer', choices=['oneformer', 'segformer'],
    #                     help='specify the semantic branch model')
    parser.add_argument('--gta', default=False, action='store_true')
    parser.add_argument('--save_original_seg', default=False, action='store_true',
                        help='whether to save original segmentation')
    parser.add_argument('--save_gt', default=False, action='store_true',
                        help='whether to save ground truth')
    parser.add_argument('--light_mode', default=False, action='store_true', help='use light mode')
    parser.add_argument('--prob_threshold', type=float, default=0.0, help='probability to update new classes')
    parser.add_argument('--data_idx', type=int, default=1)
    parser.add_argument('--data_fold', type=int, default=1)
    parser.add_argument('--train_set', default=False, action='store_true')
    parser.add_argument('--seg_only', default=False, action='store_true')

    args = parser.parse_args()
    return args


def main(rank, args):
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_path).to(rank)

    mask_branch_model = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=128 if args.dataset_sam == 'foggy_driving' else 64,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
        output_mode='coco_rle',
    )
    print('[Model loaded] Mask branch (SAM) is loaded.')

    semantic_branch_model, dataloader = seg(args)

    data_num = len(dataloader)
    print('[Data loaded] ', data_num)

    # clip_model, _, clip_processor = open_clip.create_model_and_transforms('convnext_large_d_320',
    #                                                                       pretrained='laion2b_s29b_b131k_ft_soup',
    #                                                                       device='cuda')
    clip_model, _, clip_processor = open_clip.create_model_and_transforms('convnext_large_d_320',
                                                                          pretrained='./checkpoints/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/open_clip_pytorch_model.bin',
                                                                          device='cuda')

    print('[Model loaded] CLIP & BLIP (your own segmentor) is loaded.')

    print('[SSA start] model inference starts.')

    image_dir = os.path.join(args.out_dir, 'images')
    vis_dir = os.path.join(args.out_dir, 'visualization')
    label_dir = os.path.join(args.out_dir, 'labels')
    json_dir = os.path.join(args.out_dir, 'json')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    for i, data in enumerate(dataloader):
        print('[Runing] ', i, '/', data_num, ' ', 'on rank ', rank, '/', args.world_size)
        if args.dataset_sam == 'cityscapes' or args.dataset_sam == 'foggy_driving':
            id2label = CONFIG_CITYSCAPES_ID2LABEL
        elif args.dataset_sam == 'mapillary_openset':
            id2label_mapi = config_mapillary_id2label
            id2label = CONFIG_CITYSCAPES_ID2LABEL
        else:
            raise NotImplementedError()
        with torch.no_grad():
            semantic_segment_anything_inference_clip(args.out_dir, data=data, save_img=args.save_img,
                                                     semantic_branch_model=semantic_branch_model,
                                                     mask_branch_model=mask_branch_model,
                                                     id2label=id2label,
                                                     clip_model=clip_model,
                                                     prob_threshold=args.prob_threshold
                                                     )
    if args.eval_sam and rank == 0:
        assert args.gt_path is not None
        eval_pipeline(args.gt_path, args.out_dir, args.dataset_sam)


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    if args.world_size > 1:
        mp.spawn(main, args=(args,), nprocs=args.world_size, join=True)
    else:
        main(0, args)
