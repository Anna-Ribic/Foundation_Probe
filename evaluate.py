import hydra
from omegaconf import DictConfig, OmegaConf
from src.tasks.vos import VOSTask
from hydra.utils import instantiate

@hydra.main(config_path="./configs", config_name="defaults.yaml", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print("Loading model and probe...")
    model = instantiate(cfg.models)
    probe = instantiate(cfg.probes)
    print(f"Initialized model: {model.name}"
          f" with probe: {probe.name}")

    print("Instantiating dataset...")
    dataset = instantiate(cfg.datasets)
    print(f"Initialized dataset: {dataset.name}")

    print("Initializing Task...")
    task =  instantiate(cfg.tasks)
    print(f"Initialized task: {task.name}")

    print("Running evaluation...")
    task.evaluate(model, probe, dataset)


if __name__ == "__main__":
    main()
