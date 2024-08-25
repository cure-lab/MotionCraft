export CUDA_VISIBLE_DEVICES=4

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python ./tools/s2g_test.py \
       ./configs/stmogen/S2G_Beats2_no_face_loss_025b.py \
       ./outputs/S2G_t2m_no_face_loss_l8_latentdim128_ffsize512/epoch_24.pth \
        --out=./outputs/S2G_t2m_no_face_loss_l8_latentdim128_ffsize512/ \
        --beats2_args ./mogen/datasets/EMAGE_2024/configs/st_mogen_emage.yaml \
        --deterministic >>./outputs/S2G_t2m_no_face_loss_l8_latentdim128_ffsize512/eval_epoch_24.log 2>&1

# baselines
# python ./tools/s2g_test_mcm.py \
#         outputs/mcm_beats2_s2g/mcm_s2g_beats2.py \
#         --work-dir=./outputs/mcm_beats2_s2g \
#         ./outputs/mcm_beats2_s2g/epoch_24.pth \
#         --out=./outputs/mcm_beats2_s2g/ \
#         --beats2_args "./mogen/datasets/EMAGE_2024/configs/st_mogen_emage.yaml" \
#         --deterministic >>./outputs/mcm_beats2_s2g/eval_epoch_24.log 2>&1

