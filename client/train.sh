#/bin/bash

for i in {1..5}
do
    echo "Execution number: $i"

    python ./train.py --algo ppo --env SmallSizeLeague-v1 \
        --eval-freq 10000 --eval-episodes 50 --n-eval-envs 10 \
        --save-freq 100000 --yaml-file ppo.yml \
        --tensorboard-log tmp/stable-baselines/ \
        --device cuda --vec-env subproc \
        &> executions/debug_client.txt 2>&1

    rm executions/*

    python ./train.py --algo sac --env SmallSizeLeague-v1 \
        --eval-freq 10000 --eval-episodes 50 --n-eval-envs 10 \
        --save-freq 100000 --yaml-file sac.yml \
        --tensorboard-log tmp/stable-baselines/ \
        --device cuda --vec-env subproc \
        &> executions/debug_client.txt 2>&1

    rm executions/*
done
