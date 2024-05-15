#!/bin/bash

mkdir -p train_output

for n in 0 50 100
do
    for i in {1..20}
    do
        # SAC
        python -m train --device 0 --seed $i --work-dir train_output/baseline \
                        --domain-name ContinuousCartpole-v0 --random-reward \
                        --agent baseline --noisy-observation --noisy-dims $n

        # Myopic PSEs
        python -m train --device 0 --seed $i --work-dir train_output/myopic \
                        --domain-name ContinuousCartpole-v0 --random-reward \
                        --agent pse --noisy-observation --noisy-dims $n
        
        # Myopic PSEs (multiple)
        python -m train --device 0 --seed $i --work-dir train_output/multi-myopic \
                        --domain-name ContinuousCartpole-v0 --random-reward \
                        --agent pse --multi-positive --noisy-observation --noisy-dims $n

        # Myopic DBC
        python -m train --device 0 --seed $i --work-dir train_output/bisim \
                        --domain-name ContinuousCartpole-v0 --random-reward \
                        --agent bisim --noisy-observation --noisy-dims $n
    done
done
