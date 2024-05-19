TIMESTAMP=`date "+%Y-%m-%d_%H-%M-%S"`
echo $TIMESTAMP
mkdir -p log

# RDSR x2 iso 
# CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=6 python train_rdsr_disc_v2.py \
#                 --cfg_path cfg/config_rdsr_x2_iso.json >> ./log/Dadn_train_$TIMESTAMP.log 2>&1 &

# RDSR x2 aniso 
# CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=1 python train_rdsr_disc_v83.py \
#                 --cfg_path cfg/config_rdsr_x2_aniso.json 
                #>> ./log/Dadn_train_$TIMESTAMP.log 2>&1 &


# RDSR x2 aniso - Phu
CUDA_VISIBLE_DEVICES=1 python train_rdsr_disc_v83_phu.py \
                --cfg_path cfg/config_rdsr_x2_aniso_phu.json 
                #>> ./log/Dadn_train_$TIMESTAMP.log 2>&1 &

# RDSR x4 iso
# CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=4 python train_rdsr_disc_v44.py \
#                 --cfg_path cfg/config_rdsr_x4_iso.json  >> ./log/Dadn_train_$TIMESTAMP.log 2>&1 &

# RDSR x4 aniso         
# CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=0 python train_rdsr_disc_v43.py \
#                 --cfg_path cfg/config_rdsr_x4_aniso.json >> ./log/Dadn_train_$TIMESTAMP.log 2>&1 &

# RDSR gt x2 aniso
# CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=3 python train_gt.py \
#                 --cfg_path cfg/config_rdsr_x2_aniso_gt.json >> ./log/Dadn_train_$TIMESTAMP.log 2>&1 &

# RDSR x2 aniso train dr vector only
# CUDA_VISIBLE_DEVICES=1 python train_rdsr_disc_v84.py \
#                 --cfg_path cfg/config_rdsr_x2_aniso_dr_only.json 
                #>> ./log/Dadn_train_$TIMESTAMP.log 2>&1 &

# RDSR x2 aniso add dn discriminator
# CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=1 python train_rdsr_disc_v85.py \
#                 --cfg_path cfg/config_rdsr_x2_aniso_dn_gan.json >> ./log/Dadn_train_$TIMESTAMP.log 2>&1 &

# RDSR x2 aniso train dr vector only + dn discriminator
# CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=6 python train_rdsr_disc_v845.py \
#                 --cfg_path cfg/config_rdsr_x2_aniso_dr_only_dn_gan.json >> ./log/Dadn_train_$TIMESTAMP.log 2>&1 &

# not verify
# RDSR x2 iso train dr vector only
# CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=2 python train_rdsr_disc_v21.py \
#                 --cfg_path cfg/config_rdsr_x2_iso_dr_only.json >> ./log/Dadn_train_$TIMESTAMP.log 2>&1 &

# RDSR x2 iso add dn discriminator
# CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=1 python train_rdsr_disc_v22.py \
#                 --cfg_path cfg/config_rdsr_x2_iso_dn_gan.json >> ./log/Dadn_train_$TIMESTAMP.log 2>&1 &

# RDSR x2 iso add dn discriminator DOE
# CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=6 python train_rdsr_disc_v22.py \
#                 --cfg_path cfg/config_rdsr_x2_iso_dn_gan_doe1.json >> ./log/Dadn_train_$TIMESTAMP.log 2>&1 &
