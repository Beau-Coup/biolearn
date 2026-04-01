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

`uv run scripts/train_scratch.py --system hill --boundary-samples 64 --regularizer 1e-1 --final-reg 1e-1 --importance-sample --loss slackrelu  --lr 5e-3 --num-epochs 500 --n-samples 128 --num-initializations 128 --num-instantiations 12 --runs 1`

`uv run scripts/train_scratch.py --system hill --boundary-samples 64 --regularizer 1e-1 --final-reg 1e-1 --no-importance-sample --loss slackrelu  --lr 5e-3 --num-epochs 500 --n-samples 128 --num-initializations 128 --num-instantiations 12 --runs 1`

`uv run scripts/train_scratch.py --system hill --boundary-samples 64 --regularizer 1e2 --final-reg 1e2 --importance-sample --loss slackrelu  --lr 5e-3 --num-epochs 500 --n-samples 128 --num-initializations 128 --num-instantiations 12 --runs 1`

In order: 
`hill_2a9a03a_027ca6a6`
`hill_2a9a03a_a5311b9a`
`hill_2a9a03a_ae821660`

## Laub Loomis
`uv run scripts/train_scratch.py --system laub --boundary-samples 64 --regularizer 1e-1 --final-reg 1e-1 --importance-sample --loss slackrelu --lr 2e-3 --num-epochs 700 --n-samples 256 --num-initializations 128 --num-instantiations 16 --runs 1`

laub_f5b0d1c_53eee043
`uv run scripts/train_scratch.py --system laub --boundary-samples 64 --regularizer 1e-1 --final-reg 1e-1 --importance-sample --loss slackrelu  --lr 2e-3 --num-epochs 700 --n-samples 256 --num-initializations 128 --num-instantiations 8 --runs 1`

neither: `laub_94e15f3_a7f7effd`
both: `laub_94e15f3_9d531372`
No IS: `laub_94e15f3_7f814094` doesn't change anything, almost
no residual: 

## Run quadrotor
Getting experiments now

SlackReLU: `quadrotor_918f69a_acb8a235`
ReLU: `quadrotor_918f69a_98e11cb1`
LeakyReLU: `quadrotor_918f69a_b33e43bd`

With new initializations

SlackReLU: `quadrotor_d34ce11_acb8a235`
ReLU: `quadrotor_d34ce11_98e11cb1`
LeakyReLU: `quadrotor_d34ce11_b33e43bd`

## Rerunning Hill with more importance samples

In order of run script

64 IS: `hill_94e15f3_8c0a426d`
NO IS: `hill_94e15f3_c79c2ec7`
64 IS, no MLP: `hill_94e15f3_8a126949`
100 IS: `hill_94e15f3_68bdfee0`
100 IS no mlp: `hill_94e15f3_c8b67e8e`
neither: `hill_47af3eb_bca81456`

### No MLP runs correct
100 IS: 
64 IS: `hill_47af3eb_951bd046`
No IS: 

# 2000 Epochs Experiments
