uv run scripts/train_scratch.py --system quadrotor --boundary-samples 64 --regularizer 1e-1 --final-reg 1e-1 --importance-sample --loss slackrelu --lr 5e-2 --num-epochs 2000 --n-samples 256 --num-initializations 128 --num-instantiations 16 --runs 1  --num-importance-samples 100

uv run scripts/train_scratch.py --system quadrotor --boundary-samples 64 --regularizer 1e-1 --final-reg 1e-1 --importance-sample --loss relu --lr 5e-2 --num-epochs 2000 --n-samples 256 --num-initializations 128 --num-instantiations 16 --runs 1 --num-importance-samples 100

uv run scripts/train_scratch.py --system quadrotor --boundary-samples 64 --regularizer 1e-1 --final-reg 1e-1 --importance-sample --loss leakyrelu --lr 5e-2 --num-epochs 2000 --n-samples 256 --num-initializations 128 --num-instantiations 16 --runs 1 --num-importance-samples 100

