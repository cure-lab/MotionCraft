export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
CONFIG=$1
WORK_DIR=$2
GPUS=$3
PORT=${PORT:-29380}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/ctrlnet_train.py $CONFIG --work-dir=${WORK_DIR} --launcher pytorch ${@:4}

# bash ./tools/ctrlnet_train.sh ./configs/mcm/mcm_m2d_finedance.py ./outputs/mcm_finedance_m2d 8 --no-validate --seed 42 --deterministic
# bash ./tools/ctrlnet_train.sh ./configs/mcm/mcm_s2g_beats2.py ./outputs/mcm_beats2_s2g 8 --no-validate --seed 42 --deterministic

# bash ./tools/ctrlnet_train.sh ./configs/stmogen/S2G_Beats2_no_face_loss.py ./outputs/S2G_t2m_no_face_loss 8 --no-validate --seed 42 --deterministic  
# bash ./tools/ctrlnet_train.sh ./configs/stmogen/M2D_finedance_no_face_loss.py ./outputs/M2D_t2m_no_face_loss_train 8 --no-validate --seed 42 --deterministic  

# bash ./tools/ctrlnet_train.sh ./configs/stmogen/S2G_Beats2_no_face_loss_0125b.py ./outputs/S2G_t2m_no_face_loss_l4_latentdim128_ffsize512 8 --no-validate --seed 42 --deterministic  
# bash ./tools/ctrlnet_train.sh ./configs/stmogen/S2G_Beats2_no_face_loss_025b.py ./outputs/S2G_t2m_no_face_loss_l8_latentdim128_ffsize512 8 --no-validate --seed 42 --deterministic  

# bash ./tools/ctrlnet_train.sh ./configs/stmogen/M2D_finedance_no_face_loss_0125b.py ./outputs/M2D_t2m_no_face_loss_l4_latentdim128_ffsize512 8 --no-validate --seed 42 --deterministic  
# bash ./tools/ctrlnet_train.sh ./configs/stmogen/M2D_finedance_no_face_loss_025b.py ./outputs/M2D_t2m_no_face_loss_l8_latentdim128_ffsize512 8 --no-validate --seed 42 --deterministic  

# no face loss & unfreeze locally

# bash ./tools/ctrlnet_train.sh ./configs/stmogen/S2G_Beats2_no_face_loss_0125b_local_unfreeze.py ./outputs/S2G_t2m_no_face_loss_l4_latentdim128_ffsize512_local_unfreeze 8 --no-validate --seed 42 --deterministic  

# bash ./tools/ctrlnet_train.sh ./configs/stmogen/M2D_finedance_no_face_loss_0125b_local_unfreeze.py ./outputs/M2D_t2m_no_face_loss_l4_latentdim128_ffsize512_local_unfreeze 8 --no-validate --seed 42 --deterministic  
