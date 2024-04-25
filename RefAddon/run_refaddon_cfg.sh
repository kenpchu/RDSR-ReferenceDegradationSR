TIMESTAMP=`date "+%Y-%m-%d_%H-%M-%S"`
echo $TIMESTAMP
mkdir -p log

# DualSR + Ref add-on x2
CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=0 python main1.py \
                --cfg_path cfg/config_addon_div2k_x2.json 
                # >> ./log/Dadn_train_$TIMESTAMP.log 2>&1 &
