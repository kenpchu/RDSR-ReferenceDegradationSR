TIMESTAMP=`date "+%Y-%m-%d_%H-%M-%S"`
echo $TIMESTAMP
mkdir -p log

# RDSR x2 iso 
CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=6 python train_rdsr_disc_v2.py \
                --cfg_path cfg/config_rdsr_x2_iso.json

# >> ./log/Dadn_train_$TIMESTAMP.log 2>&1 &

