#!/bin/sh
PYTHON=$HOME/miniconda3/bin/python
COUNT=11
STEPS=10000000

( id=abac; "$PYTHON" dc_learn_lotsched.py --id $id --discount 0 --max-idle 0 --action base-stock:49,19,24 --steps 0 --test-runs 30 --order-scv small --parallel rotation 2>&1 | tee $id.log; ./parse-episode-log.pl <$id.log >$id.csv ) &

for i in $(seq 1 $COUNT); do
  ( id=ppo1.$i; "$PYTHON" dc_learn_lotsched.py --id $id --max-idle 0 --steps $STEPS --test-runs 30 --order-scv small --parallel --single-layer ppo 2>&1 | tee $id.log; ./parse-episode-log.pl <$id.log >$id.csv) &
done

for i in $(seq 1 $COUNT); do
  ( id=ppo2.$i; "$PYTHON" dc_learn_lotsched.py --id $id --max-idle 0 --steps $STEPS --test-runs 30 --order-scv small --parallel ppo 2>&1 | tee $id.log; ./parse-episode-log.pl <$id.log >$id.csv ) &
done

wait
