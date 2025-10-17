import inspect
import math
from datetime import datetime
from pathlib import Path

import torch

from . import ae, autoregressive, bsq

patch_models = {
    n: m for M in [ae, bsq] for n, m in inspect.getmembers(M) if inspect.isclass(m) and issubclass(m, torch.nn.Module)
}

ar_models = {
    n: m
    for M in [autoregressive]
    for n, m in inspect.getmembers(M)
    if inspect.isclass(m) and issubclass(m, torch.nn.Module)
}


def train(model_name_or_path: str, epochs: int = 5, batch_size: int = 64):
    import lightning as L
    from lightning.pytorch.loggers import TensorBoardLogger

    from .data import ImageDataset, TokenDataset

    class PatchTrainer(L.LightningModule):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def training_step(self, x, batch_idx):
            x = x.float() / 255.0 - 0.5

            x_hat, additional_losses = self.model(x)
            loss = torch.nn.functional.mse_loss(x_hat, x)
            self.log("train/loss", loss, prog_bar=True)
            for k, v in additional_losses.items():
                self.log(f"train/{k}", v)
            return loss + sum(additional_losses.values())

        def validation_step(self, x, batch_idx):
            x = x.float() / 255.0 - 0.5

            with torch.no_grad():
                x_hat, additional_losses = self.model(x)
                loss = torch.nn.functional.mse_loss(x_hat, x)
            self.log("validation/loss", loss, prog_bar=True)
            for k, v in additional_losses.items():
                self.log(f"validation/{k}", v)
            if batch_idx == 0:
                self.logger.experiment.add_images(
                    "input", (x[:64] + 0.5).clamp(min=0, max=1).permute(0, 3, 1, 2), self.global_step
                )
                self.logger.experiment.add_images(
                    "prediction", (x_hat[:64] + 0.5).clamp(min=0, max=1).permute(0, 3, 1, 2), self.global_step
                )
            return loss

        def configure_optimizers(self):
          optimizer = torch.optim.AdamW(self.parameters(), lr=5e-3)
          scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
              optimizer,
              mode="min",        # reduce when validation loss stops decreasing
              factor=0.5,        # multiply LR by 0.5
              patience=3,        # wait 3 epochs before reducing
              min_lr=5e-6,       # never go below this
          )
          return {
              "optimizer": optimizer,
              "lr_scheduler": {
                  "scheduler": scheduler,
                  "monitor": "validation/loss",  # metric to watch
                  "interval": "epoch",
                  "frequency": 1,
              },
          }

        def train_dataloader(self):
            dataset = ImageDataset("train")
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)

        def val_dataloader(self):
            dataset = ImageDataset("valid")
            return torch.utils.data.DataLoader(dataset, batch_size=4096, num_workers=4, shuffle=False)

    class AutoregressiveTrainer(L.LightningModule):
      def __init__(self, model):
          super().__init__()
          self.model = model

      def training_step(self, batch, batch_idx):
          # 🔹 Unpack if dataset returns a tuple (e.g. (tokens, _))
          x = batch[0] if isinstance(batch, (tuple, list)) else batch
          x_hat, additional_losses = self.model(x)

          # Cross-entropy loss in bits per token
          loss = (
              torch.nn.functional.cross_entropy(
                  x_hat.view(-1, x_hat.shape[-1]),
                  x.view(-1),
                  reduction="sum"
              )
              / math.log(2)
              / x.shape[0]
          )

          # Log metrics
          self.log("train/loss", loss, prog_bar=True)
          for k, v in additional_losses.items():
              self.log(f"train/{k}", v)

          return loss + sum(additional_losses.values())

      def validation_step(self, batch, batch_idx):
          # 🔹 Unpack if dataset returns a tuple
          x = batch[0] if isinstance(batch, (tuple, list)) else batch

          with torch.no_grad():
              x_hat, additional_losses = self.model(x)
              loss = (
                  torch.nn.functional.cross_entropy(
                      x_hat.view(-1, x_hat.shape[-1]),
                      x.view(-1),
                      reduction="sum"
                  )
                  / math.log(2)
                  / x.shape[0]
              )

          self.log("validation/loss", loss, prog_bar=True)
          for k, v in additional_losses.items():
              self.log(f"validation/{k}", v)

          return loss

      def configure_optimizers(self):
          return torch.optim.AdamW(self.parameters(), lr=1e-3)

      def train_dataloader(self):
          dataset = TokenDataset("train")
          return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)

      def val_dataloader(self):
          dataset = TokenDataset("valid")
          return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    class CheckPointer(L.Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            fn = Path(f"checkpoints/{timestamp}_{model_name}.pth")
            fn.parent.mkdir(exist_ok=True, parents=True)
            torch.save(model, fn)
            torch.save(model, Path(__file__).parent / f"{model_name}.pth")

    # Load or create the model
    if Path(model_name_or_path).exists():
        model = torch.load(model_name_or_path, weights_only=False)
        model_name = model.__class__.__name__
    else:
        model_name = model_name_or_path
        if model_name in patch_models:
            model = patch_models[model_name]()
        elif model_name in ar_models:
            model = ar_models[model_name]()
        else:
            raise ValueError(f"Unknown model: {model_name}")

    # Create the lightning model
    if isinstance(model, (autoregressive.Autoregressive)):
        l_model = AutoregressiveTrainer(model)
    else:
        l_model = PatchTrainer(model)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger = TensorBoardLogger("logs", name=f"{timestamp}_{model_name}")
    trainer = L.Trainer(max_epochs=epochs, logger=logger, callbacks=[CheckPointer()])
    trainer.fit(
        model=l_model,
    )


if __name__ == "__main__":
    from fire import Fire

    Fire(train)
