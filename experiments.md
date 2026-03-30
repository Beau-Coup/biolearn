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




## Hill Fixed
I added back a_u as a parameter to be trained

`uv run scripts/train_scratch.py --system hill --boundary-samples 64 --regularizer 1e-1 --final-reg 1e-1 --importance-sample --loss relu  --lr 2e-3 --num-epochs 1000 --n-samples 128 --num-initializations 512 --num-instantiations 8 --runs 1`

Run: `hill_00a4b07_7d40d792`


## Laub Loomis
`uv run scripts/train_scratch.py --system laub --boundary-samples 64 --regularizer 1e-1 --final-reg 1e-1 --importance-sample --loss slackrelu --lr 2e-3 --num-epochs 300 --n-samples 256 --num-initializations 128 --num-instantiations 16 --runs 1`

laub_f5b0d1c_53eee043
