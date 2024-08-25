_base_ = ['./_base_/datasets/motionx_bs128.py']

# checkpoint saving
checkpoint_config = dict(interval=1)

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# optimizer
optimizer = dict(type='Adam', lr=2e-4)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[10])
runner = dict(type='EpochBasedRunner', max_epochs=24)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])



# model settings
model = dict(type='MotionDiffusion',
             loss_recon=dict(type='MSELoss', loss_weight=1, reduction='none'),
             diffusion_train=dict(
                 beta_scheduler='linear',
                 diffusion_steps=1000,
                 model_mean_type='start_x',
                 model_var_type='fixed_large',
             ),
             diffusion_test=dict(
                 beta_scheduler='linear',
                 diffusion_steps=1000,
                 model_mean_type='start_x',
                 model_var_type='fixed_large',
                 respace='15,15,8,6,6',
             ),
            #  inference_type='ddpm',
             inference_type='gt',
             loss_reduction='batch')
