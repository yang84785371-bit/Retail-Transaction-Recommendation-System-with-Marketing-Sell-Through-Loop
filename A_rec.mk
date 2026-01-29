data/raw/online_retail_II.xlsx
→ feature/build_processed.py
→ data/processed_v3/interactions.csv + data/processed_v3/sequences.csv + data/processed_v3/item_vocab.csv
→ feature/build_splits.py（读 sequences.csv）
→ data/processed_v3/train_next.csv + data/processed_v3/val_next.csv
→ train.py（读 train_next/val_next + item_vocab）
→ （训练时用）model/gru_rec.py 或 model/transformer_rec.py
→ outputs/<run_name>/config.json + outputs/<run_name>/metrics.jsonl + outputs/<run_name>/best.pt
→ eval_baselines.py（可并行/对照）
→ 输出基线对比结果（通常在控制台或写到 outputs，取决于你实现）