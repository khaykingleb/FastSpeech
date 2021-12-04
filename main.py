from pathlib import Path
import argparse
import json
import wandb

import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import DataLoader, Subset, random_split
import torch

from tts.collate_fn import LJSpeechCollator
from tts.spectrogram import MelSpectrogram
from tts.datasets import LJSpeechDataset
from tts.models import FastSpeech
from tts.loss import FastSpeechMSELoss
from tts.aligner import GraphemeAligner
from tts.vocoders import WaveGlow
from tts.trainer import *
from tts.utils import *


def main(config) -> None:
    if config["logger"]["use_wandb"]:
        wandb.init(project=config["logger"]["wandb_project_name"])

    seed_everything(config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if config["main"]["verbose"]:
        print(f"The training process will be performed on {device}.")
    
    if config["main"]["verbose"]:
        print("Downloading and splitting the data.")

    dataset = LJSpeechDataset(config["data"]["path_to_data"])
    train_size = int(config["trainer"]["train_ratio"] * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(config["main"]["seed"])
    )

    train_dataset = Subset(train_dataset, np.arange(config["trainer"]["batch_size"])) \
        if config["main"]["overfit"] is True else train_dataset

    train_dataloader = DataLoader(
        train_dataset, 
        collate_fn=LJSpeechCollator(),
        batch_size=config["trainer"]["batch_size"], 
        num_workers=config["main"]["num_workers"]
    )

    val_dataloader = DataLoader(
        val_dataset,
        collate_fn=LJSpeechCollator(),
        batch_size=config["trainer"]["batch_size"],
        num_workers=config["main"]["num_workers"]
    )

    if config["main"]["verbose"]:
        print("Initializing the vocoder, acoustic model, optimizer and lr_scheduler.")

    vocoder = WaveGlow().eval().to(device)
    aligner = GraphemeAligner(config).eval().to(device)
    melspectrogramer = MelSpectrogram(config).to(device)

    model = FastSpeech(config).to(device)
    criterion = FastSpeechMSELoss()

    trainable_params = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = init_obj(config["optimizer"], torch.optim, trainable_params)
    lr_scheduler = init_obj(config["lr_scheduler"], torch.optim.lr_scheduler, optimizer)

    if config["pretrained_model"]["load_model"]:
        if config["main"]["verbose"]:
            print("Downloading the pretrained model.")
        checkpoint = torch.load(config["pretrained_model"]["checkpoint_path"])
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optim_state_dict"])
    
    if config["logger"]["use_wandb"]:
        wandb.watch(model)
        
    train(
        config=config, 
        model=model, 
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        criterion=criterion,
        vocoder=vocoder,
        aligner=aligner,
        melspectrogramer=melspectrogramer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader if config["main"]["overfit"] is False else train_dataloader,
        device=device
    )

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Text-to-Speech Template")

    argparser.add_argument("-c",
                           "--config",
                           type=str,
                           required=True,
                           help="Config file path")

    args = argparser.parse_args()
    config_path = Path(args.config)
    with config_path.open("r") as file:
        config = json.load(file)

    main(config)
