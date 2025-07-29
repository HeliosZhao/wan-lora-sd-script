python -m networks.wan_extract_lora \
--model_org /home/lezhao/lezhao/sana-video/checkpoints/Wan14BT2VFusioniX/WanT2V_MasterModel.safetensors \
--model_tuned /home/lezhao/lezhao/sana-video/checkpoints/Wan14BT2VFusioniX/Wan14BT2VFusioniX_fp16_.safetensors \
--save_to output/fusionx_nop_noh_r32.safetensors \
--device cuda --dim 32

python -m networks.wan_merge_lora \
--models output/fusionx_nop_noh_r32.safetensors \
--flux_model /home/lezhao/lezhao/sana-video/checkpoints/Wan14BT2VFusioniX/WanT2V_MasterModel.safetensors \
--ratios 1.0 \
--save_to output/merged_fusionx_nop_noh_r32.safetensors
