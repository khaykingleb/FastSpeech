from pathlib import Path
import argparse
import wandb
import json

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from torch.utils.data import DataLoader, random_split
import torch

from tts.collate_fn import LJSpeechCollator
from tts.datasets import LJSpeechDataset
from tts.utils import *



def main(config) -> None:
    seed_everything(config["main"]["seed"])

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if config["main"]["verbose"]:
        print(f"The training process will be performed on {DEVICE}.")
    
    DATA_DIR = "Text-to-Speech/data" if config["main"]["use_colab"] is True else "./data"
    full_dataset = LJSpeechDataset(DATA_DIR)

    train_size = int(config["trainer"]["train_ratio"] * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, test_size], 
        generator=torch.Generator().manual_seed(config["main"]["seed"])
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=config["trainer"]["batch_size"], collate_fn=LJSpeechCollator()
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config["trainer"]["batch_size"], collate_fn=LJSpeechCollator()
    )
    if config["main"]["verbose"]:
        print("Data is downloaded and split.")




    #if config["main"]["use_wandb"]:
        #wandb.init(project=config["main"]["wandb_project_name"])
        #wandb.watch(model)
        

    



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