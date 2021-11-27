import json
import wandb
import argparse
from pathlib import Path

import torch

from tts.utils import *



def main(config):
    seed_everything(config)

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if config["main"]["verbose"]:
        print(f"The training process will be performed on {DEVICE}.")
    
    if config["main"]["use_wandb"]:
        

    



if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Text-to-Speech Template")

    argparser.add_argument("-c",
                           "--config",
                           type=str,
                           required=True,
                           help="Config file path")

    args = argparser.parse_args()
    config_path = Path(args.config)
    with config_path.open("rt") as file:
        config = json.load(file)

    main(config)