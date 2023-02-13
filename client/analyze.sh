#/bin/bash

python ./scripts/plot_train.py -a ppo -e SmallSizeLeague-v2 -y reward -f logs/ -w 500 -x steps
python ./scripts/plot_train.py -a sac -e SmallSizeLeague-v2 -y reward -f logs/ -w 500 -x steps


python ./scripts/all_plots.py \
    --algos ppo sac --env SmallSizeLeague-v2 \
    --exp-folders logs \
    --output data/data \
    --no-display

python ./scripts/plot_from_file.py\
    --input data/data.pkl --rliable \
    --latex -l PPO SAC \
    --versus --boxplot --iqm
