python eval_suite.py \
  --policy /path/to/your-ppo-checkpoint \
  --reference gpt2 \
  --dataset CarperAI/openai_summarize_tldr \
  --split "validation[:1000]" \
  --prompt_field prompt \
  --chosen_field chosen --rejected_field rejected \
  --batch_size 32 --max_length 768
