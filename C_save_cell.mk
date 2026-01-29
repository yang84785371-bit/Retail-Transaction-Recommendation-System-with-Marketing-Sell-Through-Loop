C｜动销 & 库存导向闭环（业务亮点）

data/processed_v3/interactions.csv
  -> marketing/build_sellthrough.py
  -> data/marketing/item_sellthrough.csv

(data/marketing/user_segments.csv + data/marketing/segment_top_items.csv + data/marketing/item_sellthrough.csv)
+ data/processed_v3/interactions.csv  (用于 user_hist_top_items / primary)
  -> marketing/build_user_strategy_items.py
  -> data/marketing/user_strategy_items.csv

(data/marketing/user_strategy_items.csv + data/marketing/item_sellthrough.csv)
  -> marketing/eval_strategy_sellthrough_mix.py
  -> data/marketing/strategy_sellthrough_mix.csv
