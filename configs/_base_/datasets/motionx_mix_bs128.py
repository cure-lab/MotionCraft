# dataset settings
# finedance
data_keys = ['motion', 'motion_mask', 'motion_length']
meta_keys = ['text']
text_train_pipeline = [
    dict(type='Normalize',
         mean_path='./data/datasets/motionx/humanml3d_align_mean.npy',
         std_path='./data/datasets/motionx/humanml3d_align_std.npy'),
    dict(type='Crop', crop_size=196),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect', keys=data_keys, meta_keys=meta_keys)
]
music_train_pipeline = [
    dict(type='Normalize',
         mean_path='./data/datasets/finedance/mean.npy',
         std_path='./data/datasets/finedance/std.npy'),
    dict(type='Crop', crop_size=196, stride=30),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect', keys=data_keys, meta_keys=meta_keys)
]
speech_train_pipeline = [
    dict(type='Normalize',
         mean_path='./data/datasets/beats2/PantoMatrix/mean.npy',
         std_path='./data/datasets/beats2/PantoMatrix/std.npy'),
    dict(type='Crop', crop_size=196),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect', keys=data_keys, meta_keys=meta_keys)
]

data = dict(
    samples_per_gpu=512,
    workers_per_gpu=1,
    train=dict(
        base=dict(
            type='TextMixMotionDataset'
        ),
        text=dict(
            type='RepeatDataset',
            dataset=dict(
                type='TextMotionDataset',
                dataset_name='motionx',
                data_prefix='./data',
                pipeline=text_train_pipeline,
                ann_file='humanml3d_align_train_val.txt',
                motion_dir='motion_data/smplx_322',
                text_dir='texts/semantic_labels',
                ),
            times=100
        ),
        music=dict(
            type='RepeatDataset',
            dataset=dict(
                type='FinedanceMotionDataset',
                dataset_name='finedance',
                data_prefix='./data',
                pipeline=music_train_pipeline,
                ann_file='train.txt',
                motion_dir='motion_fea163',
                text_dir='label_json',
                datasplit='cross_genre',
                music_dir='music_npy',
                ),
            times=2000
        ),
        speech=dict(
            type='RepeatDataset',
            dataset=dict(
                type='SpeechMotionDataset',
                dataset_name='beats2_null',
                data_prefix='./data',
                pipeline=speech_train_pipeline,
                ann_file='train.txt_null',
                motion_dir='motions_null',
                text_dir='texts_null',
                ann_config='mogen/datasets/EMAGE_2024/configs/st_mogen_emage.yaml'
               ),
               times=100
            ),
    ),
    test=dict(
        type='TextMotionDataset',
        dataset_name='motionx',
        data_prefix='./data',
        pipeline=text_train_pipeline,
        ann_file='humanml3d_align_test.txt',
        motion_dir='motion_data/smplx_322',
        text_dir='texts/semantic_labels',
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