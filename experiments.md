# Experimental runs and parameters

## tanh clipping but no noisy initial conditions
`uv run scripts/train_scratch.py --system quadrotor --boundary-samples 1 --regularizer 1e-1 --final-reg 1e-1 --importance-sample --loss slackrelu  --lr 5e-2 --num-epochs 1000 --n-samples 256`

$\alpha = 0.05$
