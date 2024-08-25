_base_ = ['../_base_/datasets/finedance_bs128.py']

# checkpoint saving
checkpoint_config = dict(interval=8)

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
runner = dict(type='EpochBasedRunner', max_epochs=48)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

input_feats = 322
max_seq_len = 196
latent_dim = 512
time_embed_dim = 2048
text_latent_dim = 256
ff_size = 1024
num_heads = 4
dropout = 0
dataset_name = "motionx"

joints = False
body_graph = False

# additional params to support the control net
control_cond_feats = 35
copy_blocks_num = 4
base_model = 'outputs/mcm_humanml3d_smplx/epoch_12.pth'
# condition pre encoding
condition_encode_cfg=dict(
        dataset_name='nothing',
        condition_pre_encode=False,
        condition_pre_encode_type='nothing', #init or wav2vec
        control_cond_feats=control_cond_feats,
        condition_latent_dim=latent_dim,
        condition_cfg=True,
    )
# Hand Loss Factor
hand_loss_factor = 1.0

# model settings
model = dict(type='MotionDiffusion',
             model=dict(type='MCMTransformer',
                        input_feats=input_feats,
                        max_seq_len=max_seq_len,
                        latent_dim=latent_dim,
                        time_embed_dim=time_embed_dim,
                        num_layers=8,
                        sa_block_cfg=dict(type='EfficientSelfAttention',
                                          latent_dim=max_seq_len,
                                          num_heads=num_heads,
                                          dropout=dropout,
                                          time_embed_dim=time_embed_dim),
                        ca_block_cfg=dict(type='EfficientCrossAttention',
                                          latent_dim=latent_dim,
                                          text_latent_dim=text_latent_dim,
                                          num_heads=num_heads,
                                          dropout=dropout,
                                          time_embed_dim=time_embed_dim),
                        ffn_cfg=dict(latent_dim=latent_dim,
                                     ffn_dim=ff_size,
                                     dropout=dropout,
                                     time_embed_dim=time_embed_dim),
                        text_encoder=dict(pretrained_model='clip',
                                          latent_dim=text_latent_dim,
                                          num_layers=4,
                                          num_heads=4,
                                          ff_size=2048,
                                          dropout=dropout,
                                          use_text_proj=True)),
             loss_recon=dict(type='MSELoss', loss_weight=1, reduction='none'),
             diffusion_train=dict(
                 beta_scheduler='linear',
                 diffusion_steps=1000,
                 model_mean_type='epsilon',
                 model_var_type='fixed_small',
             ),
             diffusion_test=dict(
                 beta_scheduler='linear',
                 diffusion_steps=1000,
                 model_mean_type='epsilon',
                 model_var_type='fixed_small',
             ),
             inference_type='ddpm')


data_keys = ['motion', 'motion_mask', 'motion_length', 'c']
meta_keys = ['text']
train_pipeline = [
    dict(type='Normalize',
         mean_path='./data/datasets/finedance/mean.npy',
         std_path='./data/datasets/finedance/std.npy'),
    dict(type='ContrlCrop', crop_size=196, stride=30),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect', keys=data_keys, meta_keys=meta_keys)
]


data = dict(samples_per_gpu=128,
            train=dict(dataset=dict(pipeline=train_pipeline)))