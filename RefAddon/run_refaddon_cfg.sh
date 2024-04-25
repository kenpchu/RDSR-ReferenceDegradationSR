TIMESTAMP=`date "+%Y-%m-%d_%H-%M-%S"`
echo $TIMESTAMP
mkdir -p log

# DualSR + Ref add-on x2
# CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=6 python main1.py \
#                 --cfg_path cfg/config_addon_div2k_x2.json >> ./log/Dadn_train_$TIMESTAMP.log 2>&1 &

# DualSR + Ref add-on x4
# CUDA_VISIBLE_DEVICES=6 python main1_v21.py --cfg_path cfg/config_addon_div2k_x4.json >> ./log/Dadn_train_$TIMESTAMP.log 2>&1 &

# DualSR + Ref add-on x2 reverse
# CUDA_VISIBLE_DEVICES=6 python main2.py --cfg_path cfg/config_addon_div2k_x2-reverse.json >> ./log/Dadn_train_$TIMESTAMP.log 2>&1 &


# DualSR + Ref add-on x2 random with random seed 
CUDA_VISIBLE_DEVICES=6 python main_para_loop.py --cfg_path cfg/config_addon_div2k_x2-random.json --para_path cfg/random_x2_paras_1.json 
# >> ./log/Dadn_train_$TIMESTAMP.log 2>&1 &
