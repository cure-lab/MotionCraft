import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import codecs as cs
from os.path import join as pjoin
import random
import glob

def find_files(directory, ext='npy'):
    files = glob.glob(os.path.join(directory, f'*.{ext}'))
    file_names = []
    for file in files:
        file_names.append(os.path.basename(file))
    return file_names

def smplx_clip_align(
        base_path,
        output_dir_name,
        ann_file
    ):

    id_list = []
    with open(ann_file, 'r') as f:
        for line in f.readlines():
            id_list.append(line.strip())
    id_list = [item for item in id_list if len(item) == 6]

    file_wo_mirror_list = []
    aug_file_wo_mirror_list = []
    cnt = 0
    aug_cnt = 0
    not_exists_cnt = 0

    for id in tqdm(id_list):
        motion_path = os.path.join(f'{base_path}/motion_data/smplx_322/humanml', id+'.npy')
        if not os.path.exists(motion_path):
            not_exists_cnt+=1
        if os.path.exists(motion_path) and len(id)==6:
            file_wo_mirror_list.append(motion_path)
            motion = np.load(motion_path)
            if (len(motion)) < 40:
                cnt+=1
                continue
            text_data = []
            flag = False
            with cs.open(pjoin(f'{base_path}/texts/semantic_labels/humanml', id + '.txt')) as f:
                for line in f.readlines():
                    line_split = line.strip().split('#')
                    caption = line_split[0]
                    f_tag = float(line_split[2])
                    to_tag = float(line_split[3])
                    f_tag = 0.0 if np.isnan(f_tag) else f_tag
                    to_tag = 0.0 if np.isnan(to_tag) else to_tag

                    if f_tag == 0.0 and to_tag == 0.0:
                        flag = True
                        text_data.append(caption)
                    else:
                        try:
                            n_motion = motion[int(f_tag*30) : int(to_tag*30)]
                            if (len(n_motion)) < 40:
                                cnt+=1
                                continue
                            new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + id
                            while (new_name in aug_file_wo_mirror_list):
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + id
                    
                            txt_file_path = pjoin(f'{base_path}/texts/semantic_labels/{output_dir_name}', new_name + '.txt')
                            with open(txt_file_path, 'w', encoding='utf-8') as file:
                                for line in [caption]:
                                    file.write(line + '\n')  
                            
                            motion_file_path = pjoin(f'{base_path}/motion_data/smplx_322/{output_dir_name}', new_name + '.npy')
                            np.save(motion_file_path, n_motion)

                            aug_file_wo_mirror_list.append(new_name)
                            aug_cnt += 1
                        except:
                            print(line_split)
                            print(line_split[2], line_split[3], f_tag, to_tag, id)
                            

            if flag:
                
                aug_file_wo_mirror_list.append(id)

                txt_file_path = pjoin(f'{base_path}/texts/semantic_labels/{output_dir_name}', id + '.txt')
                with open(txt_file_path, 'w', encoding='utf-8') as file:
                    for line in text_data:
                        file.write(line + '\n')  

                motion_file_path = pjoin(f'{base_path}/motion_data/smplx_322/{output_dir_name}', id + '.npy')
                np.save(motion_file_path, motion)
            
    aug_file_wo_mirror_list = ['humanml/' + item for item in aug_file_wo_mirror_list]

    all_file_path = f'{base_path}/{output_dir_name}.txt'
    with open(all_file_path, 'w', encoding='utf-8') as file:
        for line in aug_file_wo_mirror_list:
            file.write(line + '\n')  



if __name__ == "__main__":
    kind='test'
    base_path = './motionx'
    output_dir_name = f'humanml_{kind}_align'
    ann_file = f'{base_path}/humanml3d_origin_{kind}.txt'

    smplx_clip_align(
        base_path=base_path,
        output_dir_name=output_dir_name,
        ann_file=ann_file
    )

    kind='train_val'
    base_path = './motionx'
    output_dir_name = f'humanml_{kind}_align'
    ann_file = f'{base_path}/humanml3d_origin_{kind}.txt'

    smplx_clip_align(
        base_path=base_path,
        output_dir_name=output_dir_name,
        ann_file=ann_file
    )

