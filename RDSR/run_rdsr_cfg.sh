TIMESTAMP=`date "+%Y-%m-%d_%H-%M-%S"`
echo $TIMESTAMP
mkdir -p log

# RDSR x2 iso 
# CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=6 python train_rdsr_disc_v2.py \
#                 --cfg_path cfg/config_rdsr_x2_iso.json >> ./log/Dadn_train_$TIMESTAMP.log 2>&1 &

# RDSR x2 aniso 
# CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=7 python train_rdsr_disc_v83.py \
#                 --cfg_path cfg/config_rdsr_x2_aniso.json >> ./log/Dadn_train_$TIMESTAMP.log 2>&1 &

# RDSR x4 iso
# CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=3 python train_rdsr_disc_v43.py \
#                 --cfg_path cfg/config_rdsr_x4_iso.json >> ./log/Dadn_train_$TIMESTAMP.log 2>&1 &

# RDSR x4 aniso 
# CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=3 python train_rdsr_disc_v43.py \
#                 --cfg_path cfg/config_rdsr_x4_aniso.json >> ./log/Dadn_train_$TIMESTAMP.log 2>&1 &
