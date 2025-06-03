GPU=0
python full_eval.py -m360 data/360_v2 -tat data/tandt_db/tandt -db data/tandt_db/db --output_path output/official_fast_dash_fix --gpu ${GPU} --fast --dash --preset_upperbound
