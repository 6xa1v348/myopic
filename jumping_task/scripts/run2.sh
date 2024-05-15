#!/bin/bash

mkdir -p train_output/wide


for i in {1..100}
do
    # RandConv + Myopic PSEs
    python -m train --device 0 --seed $i --work-dir train_output/wide/myopic \
                    --agent myopic --grid wide --no-validation \
                    --rand-conv --kernel-size 2 --projection \
                    --lr 0.0046 --alpha 5.0 --temperature 1.0 \
                    --gamma 0. --soft-coupling-temperature 0.01

    # RandConv + Myopic PSEs (multiple)
    python -m train --device 0 --seed $i --work-dir train_output/wide/multi-myopic \
                    --agent myopic --grid wide --no-validation \
                    --rand-conv --kernel-size 2 --projection --multi-positive \
                    --lr 0.0012 --alpha 5.0 --temperature 1.0 \
                    --gamma 0. --soft-coupling-temperature 0.01
done