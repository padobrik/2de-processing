import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
import os

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
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

if __name__ == "__main__":
    # Create an instance of the U-Net model
    model = UNet(in_channels=1, out_channels=1)
    model = model.to('cuda')  # Move the model to GPU if available

    # Define a loss function and an optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create a DataLoader for efficient batching
    batch_size = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    num_epochs = 50

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))

        model.train()  # Set the model to training mode
        running_loss = 0.0

        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to('cuda')  # Move the images to GPU if available
            masks = masks.to('cuda')  # Move the masks to GPU if available

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate the loss
            loss = criterion(outputs, masks)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate and print the average loss for this epoch
        epoch_loss = running_loss / len(dataloader)
        print("Loss: {:.4f}".format(epoch_loss))
        
    # Save the trained model
    model_path = "trained_unet.pth"
    torch.save(model.state_dict(), model_path)
    
