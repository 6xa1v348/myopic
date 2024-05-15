Code for the noisy cartpole with randomized rewards experiments.

## Usage
The Dockerfile is provided for convenience.
For example, to run the experiment on the `Myopic PSEs` model with $N_m=50$, run the following command:
  ```
  python train.py --domain-name ContinuousCartpole-v0 --agent pse \
                  --device 0 --seed 0 --work-dir train_output/myopic \
                  --noisy-observation --noisy-dims 50 --random-reward
  ```

## Reproducing Results
The script for reproducing the experiment in the paper is given in ``run.sh``

## Acknowledgements
Our code is based on the code provided in Towards Robust Bisimulation Metric Learning [[code](https://github.com/metekemertas/RobustBisimulation/tree/main)].
The original code is from Deep Bisimulation for Control [[code](https://github.com/facebookresearch/deep_bisim4control)], which is CC-BY-NC 4.0 licensed.
