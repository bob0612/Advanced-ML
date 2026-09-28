"""Independent exponential moving average of model state; no optimizer updates."""
import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        if not 0. < decay < 1.:
            raise ValueError('EMA decay must be strictly between 0 and 1.')
        self.decay = decay
        self._state = {name: value.detach().clone()
                       for name, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, current in model.state_dict().items():
            average = self._state[name]
            if average.is_floating_point():
                average.mul_(self.decay).add_(current, alpha=1. - self.decay)
            else:
                average.copy_(current)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.clone() for name, value in self._state.items()}
