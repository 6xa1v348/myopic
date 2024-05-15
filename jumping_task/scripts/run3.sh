#!/bin/bash

mkdir -p train_output/wide


for i in {1..20}
do
    # RandConv
    python -m train --device 0 --seed $i --work-dir train_output3/wide/baseline \
                    --agent baseline --grid wide --no-validation \
                    --rand-conv --kernel-size 3 \
                    --lr 0.007 --alpha 0.

    # PSEs
    python -m train --device 0 --seed $i --work-dir train_output3/wide/pse \
                    --agent pse --grid wide --no-validation \
                    --rand-conv --kernel-size 3 --projection \
                    --lr 0.0026 --alpha 5.0 --temperature 0.5 \
                    --gamma 0.999 --soft-coupling-temperature 0.01

    # RandConv + Myopic PSEs
    python -m train --device 0 --seed $i --work-dir train_output/wide/myopic \
                    --agent myopic --grid wide --no-validation \
                    --rand-conv --kernel-size 3 --projection \
                    --lr 0.0046 --alpha 5.0 --temperature 1.0 \
                    --gamma 0. --soft-coupling-temperature 0.01

    # RandConv + Myopic PSEs (multiple)
    python -m train --device 0 --seed $i --work-dir train_output/wide/multi-myopic \
                    --agent myopic --grid wide --no-validation \
                    --rand-conv --kernel-size 3 --projection --multi-positive \
                    --lr 0.0012 --alpha 5.0 --temperature 1.0 \
                    --gamma 0. --soft-coupling-temperature 0.01
done