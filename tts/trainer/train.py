from tqdm.notebook import tqdm
import wandb

import torch.nn as nn
import torch

from tts.trainer.prolong_melspecs import prolong_melspecs
from tts.trainer.prepare_batch import prepare_batch


def train_epoch(
    config, 
    model, 
    optimizer,
    lr_scheduler,
    criterion,
    aligner,
    melspectrogramer,
    train_dataloader,
    device
):
    model.train()
    train_loss, counter = 0, 0
    
    for batch in train_dataloader:
        batch = prepare_batch(batch, melspectrogramer, aligner, device)

        durations_pred, melspec_pred = model(batch.tokens, batch.durations)
        durations_pred = torch.round(torch.exp(durations_pred)).float()
        melspec_pred, batch.melspec = prolong_melspecs(
            melspec_pred, batch.melspec, config, device
        )    

        loss = criterion(durations_pred, batch.durations, melspec_pred, batch.melspec)

        nn.utils.clip_grad_norm_(model.parameters(), config["trainer"]["grad_norm_clip"])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if config["trainer"]["use_lr_scheduler"]:
            lr_scheduler.step()

        train_loss += loss.item()
        counter += 1
        
    return train_loss / counter  


def validate_epoch(
    config, 
    model,
    criterion,
    vocoder,
    aligner,
    melspectrogramer,
    val_dataloader,
    device
):
    model.eval()
    val_loss, counter = 0, 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            batch = prepare_batch(batch, melspectrogramer, aligner, device)

            durations_pred, melspec_pred = model.inference(batch.tokens)
            durations_pred = torch.round(torch.exp(durations_pred)).float()
            melspec_pred, batch.melspec = prolong_melspecs(
                melspec_pred, batch.melspec, config, device
            )    
            
            loss = criterion(durations_pred, batch.durations, melspec_pred, batch.melspec)

            if config["logger"]["use_wandb"] and batch_idx % config["logger"]["log_frequency"] == 0:             
                wandb.log({"Validation loss": loss.item()})
            
            val_loss += loss.item()
            counter += 1

        if config["logger"]["use_wandb"]:
            wandb.log({
                "Predicted Spectrogram": wandb.Image(
                    melspec_pred[0, :, :].detach().cpu().numpy(), 
                    caption=batch.transcript[0].capitalize()
                ),
                "True Spectrogram": wandb.Image(
                    batch.melspec[0, :, :].detach().cpu().numpy(),
                    caption=batch.transcript[0].capitalize()
                )
            })

            wav_pred = vocoder.inference(melspec_pred[0, :, :].unsqueeze(0).detach().cpu()).squeeze()
            wandb.log({
                "Predicted Audio": wandb.Audio(
                    wav_pred.detach().cpu().numpy(), 
                    sample_rate=config["preprocessing"]["sr"], 
                    caption=batch.transcript[0].capitalize()
                ),
                "True Audio": wandb.Audio(
                    batch.waveform[0].detach().cpu().numpy(),
                    sample_rate=config["preprocessing"]["sr"], 
                    caption=batch.transcript[0].capitalize()
                )
            })
    
    return val_loss / counter


def train(
    config, 
    model, 
    optimizer,
    lr_scheduler,
    criterion,
    vocoder,
    aligner,
    melspectrogramer,
    train_dataloader,
    val_dataloader,
    device
):    
    history_val_loss = []

    for epoch in tqdm(range(config["trainer"]["num_epoch"])):

        train_loss = train_epoch(config, model, optimizer, lr_scheduler, criterion, aligner,
                                 melspectrogramer, train_dataloader, device)

        val_loss = validate_epoch(config, model, criterion, vocoder, aligner,
                                  melspectrogramer, val_dataloader, device)
        
        history_val_loss.append(val_loss)
         
        if config["logger"]["use_wandb"]:             
            wandb.log({"Train loss": train_loss}) 
        
        if val_loss <= min(history_val_loss):
            arch = type(model).__name__
            state = {
                "arch": arch,
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config
            }
            torch.save(state, config["pretrained_model"]["path_to_save"])
