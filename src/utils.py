from hydra.utils import instantiate

def setup_model_and_probe(model_cfg, probe_cfg):
    """
    Sets up the model and probe using Hydra's instantiate function.

    Args:
        model_cfg: Configuration dictionary for the model.
        probe_cfg: Configuration dictionary for the probe.

    Returns:
        model: Initialized model.
        probe: Initialized probe.
    """
    model = instantiate(model_cfg)
    probe = instantiate(probe_cfg)
    return model, probe
