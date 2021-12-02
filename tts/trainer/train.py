def train(
        config,
        model,
        optimizer,
        lr_scheduler,
        criterion,
        device,
        train_dataloader,
        val_dataloader,
        num_epoch,
        skip_oom
    ):

        