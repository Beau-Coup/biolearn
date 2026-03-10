import pystl


def get_semantics(semantics: str, dgmsr_p: int, smooth_temperature: float):
    semantics_kwargs: dict[str, object] = {}
    if semantics == "dgmsr":
        semantics_kwargs["p"] = int(dgmsr_p)
    elif semantics == "smooth":
        semantics_kwargs["temperature"] = float(smooth_temperature)
    elif semantics in {"classical", "agm"}:
        pass
    else:
        raise ValueError(
            f"Unsupported semantics {semantics!r}. "
            "Expected one of: 'dgmsr', 'smooth', 'classical', 'agm'."
        )
    semantics_impl = pystl.create_semantics(
        semantics, backend="jax", **semantics_kwargs
    )
    return semantics_impl
