set -x 

RANDOM_ID=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 6 | head -n 1)
RUN_NAME="qwen_instruct_1.5b_MATH_rloo_run_${RANDOM_ID}"

export HF_TOKEN="/workspace"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/workspace/reasoning/intellect/OpenRLHF"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 4 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 2 \
   --colocate_actor_ref \
   --pretrain Qwen/Qwen2.5-1.5B-Instruct \
   --save_path /workspace/reasoning/intellect/outputs/checkpoints/${SAVE_FOLDER} \
   --remote_rm_url /workspace/reasoning/intellect/OpenRLHF/openrlhf/reward_functions/math.py \
   --micro_train_batch_size 16 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 8 \
   --rollout_batch_size 256 \
   --n_samples_per_prompt 8 \
   --max_samples 100000 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 6000 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --init_kl_coef 0.01 \
   --prompt_data justus27/math-hendrycks-groundtruth-openrl \
   --input_key messages \
   --reward_info_key verification_info \
   --apply_chat_template \
   --normalize_reward \
   --packing_samples \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --load_checkpoint \
   --use_wandb "$WANDB_TOKEN" \
   --wandb_run_name ${RUN_NAME} \
   --advantage_estimator rloo \
   --temperature 0.8 \
   --top_p 0.95 \
   --save_steps 4 \
   --save_hf_ckpt \


# --runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}' [Install deps]
# --ref_reward_offload [Offload to CPU]
# --remote_rm_url http://localhost:5000/get_reward

# --vllm_sync_backend nccl (Only for multi-nodes with vLLM 0.6.4+ or vLLM 0.4.2)