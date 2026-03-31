uv run scripts/train_scratch.py --system hill --boundary-samples 64 --regularizer 1e-1 --final-reg 1e-1 --importance-sample --loss slackrelu  --lr 5e-3 --num-epochs 500 --n-samples 128 --num-initializations 128 --num-instantiations 8 --runs 1 --num-importance-samples 64 

uv run scripts/train_scratch.py --system hill --boundary-samples 64 --regularizer 1e-1 --final-reg 1e-1 --no-importance-sample --loss slackrelu  --lr 5e-3 --num-epochs 500 --n-samples 128 --num-initializations 128 --num-instantiations 8 --runs 1

uv run scripts/train_scratch.py --system hill --boundary-samples 64 --regularizer 1e4 --final-reg 1e4 --importance-sample --loss slackrelu  --lr 5e-3 --num-epochs 500 --n-samples 128 --num-initializations 128 --num-instantiations 8 --runs 1  --num-importance-samples 64

uv run scripts/train_scratch.py --system hill --boundary-samples 64 --regularizer 1e-1 --final-reg 1e-1 --importance-sample --loss slackrelu  --lr 5e-3 --num-epochs 500 --n-samples 128 --num-initializations 128 --num-instantiations 8 --runs 1  --num-importance-samples 100

uv run scripts/train_scratch.py --system hill --boundary-samples 64 --regularizer 1e4 --final-reg 1e4 --importance-sample --loss slackrelu  --lr 5e-3 --num-epochs 500 --n-samples 128 --num-initializations 128 --num-instantiations 8 --runs 1  --num-importance-samples 100

