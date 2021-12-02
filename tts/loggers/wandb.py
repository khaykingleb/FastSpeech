import wandb


class Wandb:

    def __init__(self, wandb_config):
        #wandb.login(key=wandb_config["logger"]["wandb_key"])
        wandb.init(project=wandb_config["logger"]["wandb_project_name"])