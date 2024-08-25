motion_length=90
device="cpu"


texts=(
    "a person walks forward and stumbles."
    "a person does 2 jumping jacks."
    "the person is squatting, bending at the knees, keeping their back straight, while extending their arms." 
    )

###################### 
# STMA
###################### 
config_path="./configs/stmogen/T2M_motionx_align_Finedance_Beats2_face_no_loss_0_125b.py"
pth_path="./outputs/t2m_no_face_loss_l4_latentdim128_ffsize512/epoch_12.pth"


for text in "${texts[@]}"; do
    out_file=$(echo "$text" | tr ' ' '_')
    out_path="./samples/humanml3d/t2m_no_face_loss_l4_latentdim128_ffsize512/"

    # 构建并执行命令
    PYTHONPATH=".":$PYTHONPATH python ./tools/visualize.py \
        $config_path \
        $pth_path \
        --text "$text" \
        --motion_length $motion_length \
        --out "$out_path" \
        --device $device
done