data/processed_v3/interactions.csv
→ marketing/build_rfm.py
→ data/marketing/user_rfm.csv
→ marketing/segment_rfm.py（读 user_rfm.csv）
→ data/marketing/user_segments.csv
→ marketing/segment_item_pref.py（读 interactions.csv + user_segments.csv）
→ data/marketing/segment_item_pref.csv（或你设置的 out_path）
→ marketing/strategy_router.py（读 user_segments.csv）
→ （输出到表内）strategy 列 / 分布统计（通常打印或另存）