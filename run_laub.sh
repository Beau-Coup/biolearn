uv run scripts/train_scratch.py --system laub --boundary-samples 64 --regularizer 1e-1 --final-reg 1e-1 --importance-sample --loss slackrelu  --lr 2e-3 --num-epochs 2000 --n-samples 256 --num-initializations 128 --num-instantiations 8 --runs 1  --num-importance-samples 64

uv run scripts/train_scratch.py --system laub --boundary-samples 64 --regularizer 1e-1 --final-reg 1e-1 --no-importance-sample --loss slackrelu  --lr 2e-3 --num-epochs 2000 --n-samples 256 --num-initializations 128 --num-instantiations 8 --runs 1

uv run scripts/train_scratch.py --system laub --boundary-samples 64 --regularizer 1e2 --final-reg 1e2 --importance-sample --loss slackrelu  --lr 2e-3 --num-epochs 2000 --n-samples 256 --num-initializations 128 --num-instantiations 8 --runs 1  --num-importance-samples 64

uv run scripts/train_scratch.py --system laub --boundary-samples 64 --regularizer 1e2 --final-reg 1e2 --no-importance-sample --loss slackrelu  --lr 2e-3 --num-epochs 2000 --n-samples 256 --num-initializations 128 --num-instantiations 8 --runs 1  --num-importance-samples 64
