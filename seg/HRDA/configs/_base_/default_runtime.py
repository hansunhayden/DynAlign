# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = "/home/hansun/nas/hansun/CODE/Diffusion_Repo/SSA/scripts/seg/HRDA/work_dirs/local-basic/241122_0006_gtaHR2csHR_hrda_s1_a31c3/latest.pth"
workflow = [('train', 1)]
cudnn_benchmark = True
