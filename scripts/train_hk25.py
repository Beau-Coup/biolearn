"""Training script for the HK25 fast_produce specification using BioGNN."""

import jax
import tyro
from trainer import TrainConfig, train_one
from training_suite import run_suite

from biolearn.specifications.hk25 import FastProduce
from biolearn.tasks.hk25 import HK25

jax.config.update("jax_enable_x64", True)


def main(cfg: TrainConfig):
    key = jax.random.PRNGKey(cfg.seed)
    spec = FastProduce(
        semantics=cfg.semantics,
        dgmsr_p=cfg.dgmsr_p,
        smooth_temperature=cfg.smooth_temperature,
    )

    def run_one(seed_idx, ki):
        task = HK25()
        return train_one(task, cfg, spec, seed_idx, ki)

    run_suite(run_one, cfg.n_seeds, key=key)


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
