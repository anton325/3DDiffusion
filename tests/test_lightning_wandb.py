import os
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

class DummyDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
        self.data = torch.randn(size, 10)  # Example feature size of 10
        self.labels = torch.randint(0, 2, (size,))  # Binary classification
        print(self.labels.shape)
        self.labels = torch.linspace(0, size, size+1)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class SimpleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 1)
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.validation_step_count = 0
        self.epoch = 0

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.float().unsqueeze(1))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.validation_step_count = 0
        return loss

    def validation_step(self, batch, batch_idx):
        if self.validation_step_count == 0:
            self.epoch += 1
        print(f"\nepoch {self.epoch} Validation_step {self.validation_step_count}\n")
        self.validation_step_count += 1
        x, y = batch
        print(y)
        y_hat = self(x)
        loss = self.loss(y_hat, y.float().unsqueeze(1))
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

train_dataset = DummyDataset(size=600)
val_dataset = DummyDataset(size=200)

val_sampler = torch.utils.data.RandomSampler(val_dataset,replacement=True, num_samples=50)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=20, sampler=val_sampler)


# sampler = torch.utils.data.RandomSampler(
#                 self.val, replacement=True, num_samples=self.val_size
#             )

#         return torch.utils.data.DataLoader(
#             self.val,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             sampler=sampler,
#             pin_memory=True,
#             shuffle=False,
#         )

wandb_logger = WandbLogger(project="wandb-test-dataloaders-save", log_model="all",mode="disabled")
wandb_logger = None

checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(os.getcwd(), "checkpoints"),
    filename="{epoch}-{val_loss:.2f}",
    save_top_k=-1,  # Saves all epochs
    verbose=True,
    monitor="val_loss",
    mode="min",
)


model = SimpleModel()

if wandb_logger is not None:
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        max_epochs=10,
        log_every_n_steps=1,
        precision=32,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="value",
        num_sanity_val_steps = 0,
        accelerator="gpu",
    )
else:
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=10,
        log_every_n_steps=1,
        precision=32,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="value",
        num_sanity_val_steps = 0,
        accelerator="gpu",
    )


trainer.fit(model, train_loader, val_loader)
