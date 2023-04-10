import os
import torch
import torchvision
import numpy as np
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet50
import wandb
from functools import partial
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

class UNet_Small(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet_Small, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x3 = self.decoder(x2)
        return x3


class PASCALVOCDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx])

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        mask = mask.squeeze(dim=0)
        return image, mask

def load_voc_data(root_dir, split):
    with open(os.path.join(root_dir, 'ImageSets', 'Segmentation', f'{split}.txt')) as f:
        image_names = f.read().splitlines()

    image_paths = [os.path.join(root_dir, 'JPEGImages', f'{name}.jpg') for name in image_names]
    mask_paths = [os.path.join(root_dir, 'SegmentationClassBW', f'{name}.png') for name in image_names]

    return image_paths, mask_paths

def main(model_type='small_unet'):
    wandb.init(project="unet-pascal-voc")
    root_dir = '/content/VOCdevkit/VOC2012'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_image_paths, train_mask_paths = load_voc_data(root_dir, 'train')
    val_image_paths, val_mask_paths = load_voc_data(root_dir, 'val')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = PASCALVOCDataset(train_image_paths, train_mask_paths, transform)
    val_dataset = PASCALVOCDataset(val_image_paths, val_mask_paths, transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    if model_type == 'small_unet':
        model = UNet_Small(in_channels=3, out_channels=2).to(device)
        wandb.watch(model, log='all')
    elif model_type == 'standard_unet':
        model = UNet(n_channels=3, n_classes=2).to(device)
        wandb.watch(model, log='all')
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Choose either 'small_unet' or 'standard_unet'.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.long())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f'Epoch: {epoch}, Train Loss: {train_loss}')

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)

                masks_pred = model(images)
                loss = criterion(masks_pred, masks.long())

                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f'Epoch: {epoch}, Val Loss: {val_loss}')
        try:
          wandb.log({
              'learning rate': optimizer.param_groups[0]['lr'],
              'validation Dice': val_loss,
              'images': wandb.Image(images[0].cpu()),
              'masks': {
                  'true': wandb.Image(masks[0].float().cpu()),
                  'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
              },
              'epoch': epoch,
              "Train Loss": train_loss,
              "Val Loss": val_loss

          })
        except:
          pass

    wandb.finish()

if __name__ == '__main__':
    main(model_type='standard_unet')