import jax
import tyro
from trainer import TrainConfig, train_one
from training_suite import run_suite

from biolearn.specifications.ss_classification import PhiXorFast, PhiXorSS
from biolearn.tasks.nfc_xor import NfcXor


def _make_spec(cfg: TrainConfig):
    """Instantiate the STL specification from config."""
    kwargs = dict(
        semantics=cfg.semantics,
        dgmsr_p=cfg.dgmsr_p,
        smooth_temperature=cfg.smooth_temperature,
    )
    if cfg.spec == "phi_xor_fast":
        return PhiXorFast(**kwargs)
    if cfg.spec == "phi_xor_ss":
        return PhiXorSS(**kwargs)
    raise ValueError(f"Unknown spec: {cfg.spec!r}")


def main(cfg: TrainConfig):
    key = jax.random.PRNGKey(cfg.seed)
    key = jax.random.split(key, num=5)[-1]
    spec = _make_spec(cfg)

    def run_one(seed_idx, ki):
        model_key, train_key = jax.random.split(ki)
        task = NfcXor(model_key, layer_sizes=(2, 1))
        return train_one(task, cfg, spec, seed_idx, train_key)

    run_suite(run_one, cfg.n_seeds, key=key)


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
