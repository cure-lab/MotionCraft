export CUDA_VISIBLE_DEVICES=0

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python ./tools/m2d_test.py \
        ./configs/stmogen/M2D_finedance_no_face_loss_0125b.py \
        ./outputs/M2D_t2m_no_face_loss_l4_latentdim128_ffsize512/epoch_48.pth \
        >> ./outputs/M2D_t2m_no_face_loss_l4_latentdim128_ffsize512/eval_epoch_48.log 2>&1
