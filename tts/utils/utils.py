import random
import os
import numpy as np
import torch


def seed_everything(main_config: dict) -> None:
    seed = main_config["main"]["seed"]
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"


def init_obj(obj_dict, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj(config['param'], module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = obj_dict["type"]
        module_args = dict(obj_dict["args"])

        assert all([k not in module_args for k in kwargs]), "Overwriting kwargs given in config file is not allowed"
        module_args.update(kwargs)

        return getattr(module, module_name)(*args, **module_args)


@torch.no_grad()
def get_grad_norm(model, norm_type=2):
    parameters = model.parameters()

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = [param for param in parameters if param.grad is not None]

    total_norm = torch.norm(
        torch.stack([
            torch.norm(param.grad.detach(), norm_type).cpu() \
            for param in parameters
        ]),
        norm_type
    )

    return total_norm.item()