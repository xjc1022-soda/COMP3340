from dataset import SADataset
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule


class SADataModule(LightningDataModule):
    def __init__(self, batch_size, num_workers, drop_last=True, pin_memory=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        
    def train_dataloader(self):
        dataset = SADataset(split="train")
        return DataLoader(
            dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        dataset = SADataset(split="valid")
        return DataLoader(
            dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

if __name__ == "__main__":
    dm = SADataModule(8, 1)
    for batch in dm.train_dataloader():
        break
    print(batch)
    img, os, event = batch
    import ipdb
    ipdb.set_trace()
