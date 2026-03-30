uv run scripts/train_scratch.py --system hill --boundary-samples 64 --regularizer 1e-1 --final-reg 1e-1 --importance-sample --loss slackrelu  --lr 5e-3 --num-epochs 500 --n-samples 128 --num-initializations 128 --num-instantiations 12 --runs 1

uv run scripts/train_scratch.py --system hill --boundary-samples 64 --regularizer 1e-1 --final-reg 1e-1 --no-importance-sample --loss slackrelu  --lr 5e-3 --num-epochs 500 --n-samples 128 --num-initializations 128 --num-instantiations 12 --runs 1

uv run scripts/train_scratch.py --system hill --boundary-samples 64 --regularizer 1e2 --final-reg 1e2 --importance-sample --loss slackrelu  --lr 5e-3 --num-epochs 500 --n-samples 128 --num-initializations 128 --num-instantiations 12 --runs 1
