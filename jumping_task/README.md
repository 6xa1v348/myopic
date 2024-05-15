Code for the jumping task environment experiments.

## Environment
Please use the jumping-task environment at https://github.com/google-research/jumping-task.

## Usage
The Dockerfile is provided for convenience. For example, to run the experiment on the ``RandConv + Myopic PSEs`` model for the ``wide`` grid, run the following command:
  ```
  python train.py --aegnt myopic --rand-conv --kernel-size 2 --projection \
                  --device 0 seed 0 --work-dir train_output/myopic \
                  --grid wide --no-validation \
                  --lr 0.0046 --alpha 5.0 --temperature 1.0 \
                  --gamma 0. --soft-coupling-temperature 0.01
  ```

## Reproducing Results
The scripts for reproducing the experiments in the paper are given in the ``scripts`` folder.

## Acknowledgements
Our code is based on code provided in Contrastive behavorial similarity embeddings for generalization in reinforcement learning [[code](https://github.com/google-research/google-research/tree/master/pse/jumping_task)] which is under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0.
