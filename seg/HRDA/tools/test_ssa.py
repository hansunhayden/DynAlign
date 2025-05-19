# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Modification of config and checkpoint to support legacy models
# - Add inference mode and HRDA output flag

import argparse
import os
import math

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

from ..dammseg.apis import multi_gpu_test, single_gpu_test
from ..dammseg.datasets import build_dataloader, build_dataset
from ..dammseg.models import build_segmentor


def update_legacy_cfg(cfg):
    # The saved json config does not differentiate between list and tuple
    cfg.data.test.pipeline[1]['img_scale'] = tuple(
        cfg.data.test.pipeline[1]['img_scale'])
    cfg.data.val.pipeline[1]['img_scale'] = tuple(
        cfg.data.val.pipeline[1]['img_scale'])
    # Support legacy checkpoints
    if cfg.model.decode_head.type == 'UniHead':
        cfg.model.decode_head.type = 'DAFormerHead'
        cfg.model.decode_head.decoder_params.fusion_cfg.pop('fusion', None)
    if cfg.model.type == 'MultiResEncoderDecoder':
        cfg.model.type = 'HRDAEncoderDecoder'
    if cfg.model.decode_head.type == 'MultiResAttentionWrapper':
        cfg.model.decode_head.type = 'HRDAHead'
    cfg.model.backbone.pop('ema_drop_path_rate', None)
    return cfg

from argparse import Namespace

def daformer_args(args_ssa):
    args = Namespace()
    args.test_root = args_ssa.seg_dir
    print("*******SEGMENTATION MODEL", args.test_root)
    # TODO
    # print("***************", args.test_root)
    # print("***************", args_ssa.seg_dir)
    args_ssa.seg_dir = os.path.basename(args_ssa.seg_dir)
    if args_ssa.seg_config:
        args.config = args_ssa.seg_config
    else:
        args.config = os.path.join(args.test_root, args_ssa.seg_dir + '.py')
    # print("***************CONFIG FILE", args.config)
    # TODO
    if not os.path.exists(args.config):
        args.config = os.path.join(args.test_root, args_ssa.seg_dir + '.json')
    # print("***************CONFIG FILE", args.config)

    # TODO
    if args_ssa.seg_checkpoint:
        args.checkpoint = args_ssa.seg_checkpoint
    else:
        args.checkpoint = os.path.join(args.test_root, "latest.pth")
    print("***************SEG CHECKPOINT", args.checkpoint)
    args.show_dir = os.path.join(args.test_root, 'pred')

    args.aug_test = False
    args.inference_mode = 'same'
    args.test_set = False
    args.hrda_out = ''
    # parser.add_argument('--out', help='output result file in pickle format')
    args.format_only = False
    args.eval = 'mIoU'
    args.gpu_collect = False
    # parser.add_argument(
    #     '--tmpdir',
    #     help='tmp directory used for collecting results from multiple '
    #     'workers, available when gpu_collect is not specified')
    # parser.add_argument(
    #     '--options', nargs='+', action=DictAction, help='custom options')
    # parser.add_argument(
    #     '--eval-options',
    #     nargs='+',
    #     action=DictAction,
    #     help='custom options for evaluation')
    args.launcher = 'none'
    args.opacity = 1.0
    args.local_rank = 0
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    args.out = None
    args.tmpdir = None
    args.options = None
    args.eval_options = None

    args.train_set = args_ssa.train_set

    return args


def main():
    args = daformer_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    cfg = update_legacy_cfg(cfg)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    if args.inference_mode == 'same':
        # Use pre-defined inference mode
        pass
    elif args.inference_mode == 'whole':
        print('Force whole inference.')
        cfg.model.test_cfg.mode = 'whole'
    elif args.inference_mode == 'slide':
        print('Force slide inference.')
        cfg.model.test_cfg.mode = 'slide'
        crsize = cfg.data.train.get('sync_crop_size', cfg.crop_size)
        cfg.model.test_cfg.crop_size = crsize
        cfg.model.test_cfg.stride = [int(e / 2) for e in crsize]
        cfg.model.test_cfg.batched_slide = True
    else:
        raise NotImplementedError(args.inference_mode)

    if args.hrda_out == 'LR':
        cfg['model']['decode_head']['fixed_attention'] = 0.0
    elif args.hrda_out == 'HR':
        cfg['model']['decode_head']['fixed_attention'] = 1.0
    elif args.hrda_out == 'ATT':
        cfg['model']['decode_head']['debug_output_attention'] = True
    elif args.hrda_out == '':
        pass
    else:
        raise NotImplementedError(args.hrda_out)

    if args.test_set:
        for k in cfg.data.test:
            if isinstance(cfg.data.test[k], str):
                cfg.data.test[k] = cfg.data.test[k].replace('val', 'test')

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(
        model,
        args.checkpoint,
        map_location='cpu',
        revise_keys=[(r'^module\.', ''), ('model.', '')])
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE

    efficient_test = False
    if args.eval_options is not None:
        efficient_test = args.eval_options.get('efficient_test', False)

    if not distributed:
        print("**************single_gpu_test" )
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  efficient_test, args.opacity)
    else:
        print("**************multi_gpu_test" )
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect, efficient_test)

    # print("**********************TEST", len(outputs), outputs[0].shape)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            res = dataset.evaluate(outputs, args.eval, **kwargs)
            print([k for k, v in res.items() if 'IoU' in k])
            print([round(v * 100, 1) for k, v in res.items() if 'IoU' in k])

# TODO
def seg_model(args_ssa):
    args = daformer_args(args_ssa)

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    cfg = update_legacy_cfg(cfg)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    if args.inference_mode == 'same':
        # Use pre-defined inference mode
        pass
    elif args.inference_mode == 'whole':
        print('Force whole inference.')
        cfg.model.test_cfg.mode = 'whole'
    elif args.inference_mode == 'slide':
        print('Force slide inference.')
        cfg.model.test_cfg.mode = 'slide'
        crsize = cfg.data.train.get('sync_crop_size', cfg.crop_size)
        cfg.model.test_cfg.crop_size = crsize
        cfg.model.test_cfg.stride = [int(e / 2) for e in crsize]
        cfg.model.test_cfg.batched_slide = True
    else:
        raise NotImplementedError(args.inference_mode)

    if args.hrda_out == 'LR':
        cfg['model']['decode_head']['fixed_attention'] = 0.0
    elif args.hrda_out == 'HR':
        cfg['model']['decode_head']['fixed_attention'] = 1.0
    elif args.hrda_out == 'ATT':
        cfg['model']['decode_head']['debug_output_attention'] = True
    elif args.hrda_out == '':
        pass
    else:
        raise NotImplementedError(args.hrda_out)

    if args.test_set:
        for k in cfg.data.test:
            if isinstance(cfg.data.test[k], str):
                cfg.data.test[k] = cfg.data.test[k].replace('val', 'test')

    # TODO
    if args.train_set:
        for k in cfg.data.test:
            if isinstance(cfg.data.test[k], str):
                cfg.data.test[k] = cfg.data.test[k].replace('val', 'train')

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    # print("**********************", cfg)
    # TODO
    cfg.data.test.data_root = args_ssa.data_dir
    print("=============cfg_test", cfg.data.test)
    dataset = build_dataset(cfg.data.test)
    # dataset = dataset[:10]
    # print(len(dataset))
    # print(dataset)
    # print(dataset.img_infos)
    data_num = len(dataset)
    fold_num = math.ceil(data_num/args_ssa.data_fold)
    idx1 = (args_ssa.data_idx-1)*fold_num
    idx2 = min(idx1 + fold_num, data_num)
    print('[Data range] ', idx1, ' ', idx2)
    dataset.img_infos = dataset.img_infos[idx1:idx2]
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(
        model,
        args.checkpoint,
        map_location='cpu',
        revise_keys=[(r'^module\.', ''), ('model.', '')])
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE

    efficient_test = False
    if args.eval_options is not None:
        efficient_test = args.eval_options.get('efficient_test', False)

    if not distributed:
        print("**************single_gpu_test" )
        model = MMDataParallel(model, device_ids=[0])
        # outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
        #                           efficient_test, args.opacity)
    else:
        print("**************multi_gpu_test" )
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        # outputs = multi_gpu_test(model, data_loader, args.tmpdir,
        #                          args.gpu_collect, efficient_test)

    return model, data_loader

    # print("**********************TEST", len(outputs), outputs[0].shape)
    #
    # rank, _ = get_dist_info()
    # if rank == 0:
    #     if args.out:
    #         print(f'\nwriting results to {args.out}')
    #         mmcv.dump(outputs, args.out)
    #     kwargs = {} if args.eval_options is None else args.eval_options
    #     if args.format_only:
    #         dataset.format_results(outputs, **kwargs)
    #     if args.eval:
    #         res = dataset.evaluate(outputs, args.eval, **kwargs)
    #         print([k for k, v in res.items() if 'IoU' in k])
    #         print([round(v * 100, 1) for k, v in res.items() if 'IoU' in k])



if __name__ == '__main__':
    main()
