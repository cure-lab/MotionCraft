# export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
CONFIG=$1
WORK_DIR=$2
GPUS=$3
PORT=${PORT:-28300}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --work-dir=${WORK_DIR} --launcher pytorch ${@:4}

# bash ./tools/dist_train.sh ./configs/finemogen/finemogen_t2m_smplx.py ./outputs/finemogen_humanml3d_smplx 8 --no-validate --seed 42 --deterministic
# bash ./tools/dist_train.sh ./configs/motiondiffuse/motiondiffuse_t2m_smplx.py ./outputs/motiondiffuse_humanml3d_smplx 8 --no-validate --seed 42 --deterministic
# bash ./tools/dist_train.sh ./configs/mdm/mdm_t2m_smplx.py ./outputs/mdm_humanml3d_smplx 8 --no-validate --seed 42 --deterministic

# bash ./tools/dist_train.sh ./configs/stmogen/T2M_motionx_align_Finedance_Beats2_face_no_loss.py ./outputs/t2m_no_face_loss 8 --no-validate --seed 42 --deterministic
# bash ./tools/dist_train.sh ./configs/stmogen/T2M_motionx_align_Finedance_Beats2_face_no_loss_0_125b.py ./outputs/t2m_no_face_loss_l4_latentdim128_ffsize512 8 --no-validate --seed 42 --deterministic
# bash ./tools/dist_train.sh ./configs/stmogen/T2M_motionx_align_Finedance_Beats2_face_no_loss_0_25b.py ./outputs/t2m_no_face_loss_l8_latentdim128_ffsize512 8 --no-validate --seed 42 --deterministic
