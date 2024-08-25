export CUDA_VISIBLE_DEVICES=0

# # finemogen humanml3d smplx
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python ./tools/test.py ./configs/finemogen/finemogen_t2m_smplx.py --work-dir=./outputs/finemogen_humanml3d_smplx ./outputs/finemogen_humanml3d_smplx/epoch_12.pth --out=./outputs/finemogen_humanml3d_smplx/res_12.json

# # motiondiffuse humanml3d smplx
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python ./tools/test.py ./configs/motiondiffuse/motiondiffuse_t2m_smplx.py --work-dir=./outputs/motiondiffuse_humanml3d_smplx ./outputs/motiondiffuse_humanml3d_smplx/epoch_12.pth --out=./outputs/motiondiffuse_humanml3d_smplx/res_12.json

# # mdm humanml3d smplx
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python ./tools/test.py ./configs/mdm/mdm_t2m_smplx.py --work-dir=./outputs/mdm_humanml3d_smplx ./outputs/mdm_humanml3d_smplx/epoch_12.pth --out=./outputs/mdmhumanml3d_smplx/res_12.json

# # mcm humanml3d smplx
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python ./tools/test.py ./configs/mcm/mcm_t2m_smplx.py --work-dir=./outputs/mcm_humanml3d_smplx ./outputs/mcm_humanml3d_smplx/epoch_12.pth --out=./outputs/mcm_humanml3d_smplx/res_12.json

# gt test
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python ./tools/test.py ./configs/gt.py --work-dir=./outputs/gt ./outputs/gt/gt.pth --out=./outputs/gt/res.json

# ours
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python ./tools/test.py ./configs/stmogen/T2M_motionx_align_Finedance_Beats2_face_no_loss_0_125b.py --work-dir=./outputs/t2m_no_face_loss_l4_latentdim128_ffsize512 ./outputs/t2m_no_face_loss_l4_latentdim128_ffsize512/epoch_12.pth --out=./outputs/t2m_no_face_loss_l4_latentdim128_ffsize512/res_12.json
