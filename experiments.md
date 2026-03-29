# Experimental runs and parameters

## tanh clipping but no noisy initial conditions
`uv run scripts/train_scratch.py --system quadrotor --boundary-samples 1 --regularizer 1e-1 --final-reg 1e-1 --importance-sample --loss slackrelu  --lr 5e-2 --num-epochs 1000 --n-samples 256`

$\alpha = 0.05$

## Slightly bigger initial condition region and [0.001, 1.0] parameter range
`uv run scripts/train_scratch.py --system hill --boundary-samples 1 --regularizer 1e-1 --final-reg 1e-1 --importance-sample --loss slackrelu --lr 2e-2 --num-epochs 1000 --n-samples 300 --num-initializations 64 --num-instantiations 8 --runs 1`
Does not seem to train

## fixed initial conditions
`uv run scripts/train_scratch.py --system hill --boundary-samples 1 --regularizer 1e-1 --final-reg 1e-1 --importance-sample --loss slackrelu --lr 2e-2 --num-epochs 1000 --n-samples 256 --num-initializations 64 --num-instantiations 8 --runs 1`

Does not work, 0/1 000 000

`uv run scripts/train_scratch.py --system hill --boundary-samples 64 --regularizer 1e-1 --final-reg 1e-1 --no-importance-sample --loss relu --lr 2e-2 --num-epochs 1000 --n-samples 256 --num-initializations 64 --num-instantiations 8 --runs 1`
Same


## After fixing stuff
`uv run scripts/train_scratch.py --system quadrotor --boundary-samples 128 --regularizer 1e-1 --final-reg 1e-1 --importance-sample --loss slackrelu --lr 5e-2 --num-epochs 300 --n-samples 512 --num-initializations 128 --num-instantiations 16 --runs 1`

### Figures / paper:
`uv run scripts/train_scratch.py --system quadrotor --boundary-samples 64 --regularizer 1e-1 --final-reg 1e-1 --importance-sample --loss slackrelu --lr 5e-2 --num-epochs 300 --n-samples 256 --num-initializations 128 --num-instantiations 16 --runs 1`

Model 12 has best robustness

#### Result IDs

ReLU:quadrotor_83813b1_98e11cb1 
SlackReLU: quadrotor_83813b1_acb8a235
LeakyReLU: quadrotor_83813b1_b33e43bd



