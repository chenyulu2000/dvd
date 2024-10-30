from pathlib import Path

import torch

from torch import nn, optim


class CheckpointManager:
    def __init__(
            self,
            opt,
            models,
            optimizers,
            logger,
            checkpoint_dirpath,
            last_epoch=-1,
    ):
        for model in models.values():
            if not isinstance(model, nn.Module):
                logger.error('{} is not a Module'.format(type(model).__name__))
                raise TypeError

        for optimizer in optimizers.values():
            if not isinstance(optimizer, optim.Optimizer):
                logger.error('{} is not an Optimizer'.format(type(optimizer).__name__))
                raise TypeError
        self.opt = opt
        self.logger = logger
        self.models = models
        self.optimizers = optimizers
        self.ckpt_dirpath = Path(checkpoint_dirpath)
        self.last_epoch = last_epoch
        self.init_directory()

    def init_directory(self):
        """Initialize empty checkpoint directory."""
        self.ckpt_dirpath.mkdir(parents=True, exist_ok=True)

    def step(self, epoch=None):
        """Save checkpoint if step size conditions meet. """
        if not epoch:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if self.opt.local_rank == 0 and self.opt.save_interval != -1 and self.last_epoch % self.opt.save_interval == 0:
            state_dict = {'epoch': self.last_epoch}
            for model_name, model in self.models.items():
                state_dict[model_name] = self._model_state_dict(model=model)
            for optimizer_name, optimizer in self.optimizers.items():
                state_dict[optimizer_name] = optimizer.state_dict()
            torch.save(
                obj=state_dict,
                f=self.ckpt_dirpath / f"checkpoint_{self.last_epoch}.pth",
                _use_new_zipfile_serialization=False
            )

    @staticmethod
    def _model_state_dict(model):
        """Returns state dict of model, taking care of DataParallel case."""
        if isinstance(model, nn.parallel.DistributedDataParallel):
            return model.module.state_dict()
        else:
            return model.state_dict()

    def save_best(self, ckpt_name="best"):
        state_dict = {'epoch': self.last_epoch}
        for model_name, model in self.models.items():
            state_dict[model_name] = self._model_state_dict(model=model)
        for optimizer_name, optimizer in self.optimizers.items():
            state_dict[optimizer_name] = optimizer.state_dict()
        torch.save(
            obj=state_dict,
            f=self.ckpt_dirpath / f"checkpoint_{ckpt_name}.pth",
            _use_new_zipfile_serialization=False
        )

    def update_last_epoch(self, epoch=None):
        self.last_epoch = epoch
        self.logger.info(f'Setting the epoch number to {self.last_epoch}.')
        return


def load_checkpoint(checkpoint_pthpath):
    """Given a path to saved checkpoint, load corresponding state dicts
    of model and optimizer from it. This method checks if the current
    commit SHA of codebase matches the commit SHA recorded when this
    checkpoint was saved by checkpoint manager.

    Parameters
    ----------
    checkpoint_pthpath: str or pathlib.Path
        Path to saved checkpoint (as created by ``CheckpointManager``).

    Returns
    -------
    nn.Module, optim.Optimizer
        Model and optimizer state dicts loaded from checkpoint.
    """

    if isinstance(checkpoint_pthpath, str):
        checkpoint_pthpath = Path(checkpoint_pthpath)

    # load encoder, decoder, optimizer state_dicts
    state_dicts = torch.load(checkpoint_pthpath, map_location='cpu')
    return state_dicts
