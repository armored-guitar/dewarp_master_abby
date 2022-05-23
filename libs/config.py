import hydra
import sys
from omegaconf import DictConfig
import os


def parse_config(job_name: str = "train") -> DictConfig:
    arguments = sys.argv
    if len(arguments) == 1:
        raise ValueError("Usage: python script.py <config> <overrides> both script and config should be giver")
    else:
        script_name, config_path, overrides = arguments[0], arguments[1], arguments[2:]
        if len(overrides) == 1:
            overrides = overrides[0].split(" ")
    if not os.path.isabs(config_path):
        config_path = os.path.join("../", config_path)

    config_dir = os.path.dirname(config_path)
    config_name = os.path.basename(config_path)
    with hydra.initialize(config_path=config_dir, job_name=job_name):
        opt = hydra.compose(config_name=config_name, overrides=overrides)
    return opt
