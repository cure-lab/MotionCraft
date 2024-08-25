# dataset settings
data_keys = ['motion', 'motion_mask', 'motion_length', 'c']
meta_keys = ['text', 'token']
train_pipeline = [
    dict(type='Normalize',
         mean_path='./data/datasets/beats2/PantoMatrix/mean.npy',
         std_path='./data/datasets/beats2/PantoMatrix/std.npy'),
    dict(type='Crop', crop_size=64),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect', keys=data_keys, meta_keys=meta_keys)
]

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=1,
    train=dict(type='RepeatDataset',
               dataset=dict(
                   type='SpeechMotionDataset',
                   dataset_name='beats2_null',
                   data_prefix='data',
                   pipeline=train_pipeline,
                   ann_file='train.txt_null',
                   motion_dir='motions_null',
                   text_dir='texts_null',
                   clip_feat_dir='clip_feats_null',
                   ann_config='mogen/datasets/EMAGE_2024/configs/st_mogen_emage.yaml'
               ),
               times=100),
    test=dict(type='SpeechMotionDataset',
              dataset_name='beats2',
            eval_cfg=dict(
                  shuffle_indexes=True,
                  replication_times=20,
                  replication_reduction='statistics',
                  evaluator_model=dict(
                      type='T2MContrastiveModel_SMPLX',
                      motion_encoder=dict(
                          nfeats=322,
                          vae=True,
                          num_layers=4,
                      ),
                      text_encoder=dict(
                          modelpath='./data/evaluators/smplx322/distilbert-base-uncased',
                          num_layers=4
                      ),
                      init_cfg=dict(
                          type='Pretrained',
                          checkpoint='./data/evaluators/smplx322/epoch=199.ckpt')),
                  metrics=[
                      dict(type='R Precision', batch_size=32, top_k=3),
                      dict(type='Matching Score', batch_size=32),
                      dict(type='FID', emb_scale=1.0),
                      dict(type='Diversity', num_samples=300),
                  ]),
              test_mode=True)
    )
