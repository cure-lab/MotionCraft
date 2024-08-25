# dataset settings
data_keys = ['motion', 'motion_mask', 'motion_length', 'c']
meta_keys = ['text']
train_pipeline = [
    dict(type='Normalize',
         mean_path='./data/datasets/finedance/mean.npy',
         std_path='./data/datasets/finedance/std.npy'),
    dict(type='ContrlCrop', crop_size=120, stride=30),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect', keys=data_keys, meta_keys=meta_keys)
]

data = dict(
    samples_per_gpu=512,
    workers_per_gpu=1,
    train=dict(type='RepeatDataset',
               dataset=dict(
                   type='FinedanceMotionDataset',
                   dataset_name='finedance',
                   data_prefix='./data',
                   pipeline=train_pipeline,
                   ann_file='train.txt',
                   motion_dir='motion_fea163',
                   text_dir='label_json',
                   datasplit='cross_genre',
                   music_dir='music_npy',
               ),
               times=2000),
    test=dict(type='FinedanceMotionDataset',
              dataset_name='finedance',
              data_prefix='./data',
              pipeline=train_pipeline,
              ann_file='test.txt',
              motion_dir='motion_fea163',
              text_dir='label_json',
              datasplit='cross_genre',
              music_dir='music_npy',
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
              test_mode=True))
