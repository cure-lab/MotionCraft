export CUDA_VISIBLE_DEVICES=4
device="cuda"


config_path="./configs/stmogen/S2G_Beats2_no_face_loss_025b.py"
pth_path="./outputs/S2G_t2m_no_face_loss_l8_latentdim128_ffsize512/epoch_24.pth"
out_path="./samples/beats/S2G_t2m_no_face_loss_l8_latentdim128_ffsize512-24-fs-0-hs-1.0-fixtrans/"


PYTHONPATH=".":$PYTHONPATH python ./tools/s2g_visualize.py \
    $config_path \
    $pth_path \
    --beats2_args "./mogen/datasets/EMAGE_2024/configs/emage_test_stmogen.yaml" \
    --out "$out_path" \
    --device $device \
    --repaint \
    --overlap_len 16 \
    --timestep_respacing ddim25 \
    --fix_very_first  >> ./samples/beats/S2G_t2m_no_face_loss_l8_latentdim128_ffsize512-24-fs-0-hs-1.0-fixtrans/visual.log 2>&1
