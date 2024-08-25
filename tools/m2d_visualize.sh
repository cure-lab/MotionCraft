export CUDA_VISIBLE_DEVICES=0
motion_length=120
device="cuda"

texts=(
    "198"
    "006"
    "098"
    "013"
    )
slices=(
    1 2 3 4 5 10 15 20 # 30 35 40
)

config_path="./outputs/M2D_t2m_no_face_loss_l4_latentdim128_ffsize512/M2D_finedance_no_face_loss_0125b.py"
pth_path="./outputs/M2D_t2m_no_face_loss_l4_latentdim128_ffsize512/epoch_48.pth"

for text in "${texts[@]}"; do
    for slice in "${slices[@]}"; do
        PYTHONPATH=".":$PYTHONPATH python ./tools/m2d_visualize.py \
            $config_path \
            $pth_path \
            --text "$text" \
            --slice_idx $slice \
            --motion_length $motion_length \
            --out './samples/finedance/M2D_t2m_no_face_loss_l4_latentdim128_ffsize512-48' \
            --device $device >> ./samples/finedance/M2D_t2m_no_face_loss_l4_latentdim128_ffsize512-48/visual.log 2>&1
    done
done
