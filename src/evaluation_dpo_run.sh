python eval_suite.py \
  --policy /path/to/your-dpo-checkpoint \
  --reference gpt2 \
  --dataset Anthropic/hh-rlhf \
  --split "test[:1000]" \
  --batch_size 32 --max_length 768 --seed 42
