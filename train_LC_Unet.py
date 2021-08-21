from torch.utils.data import DataLoader, Dataset
from batchgenerators.transforms.spatial_transforms import SpatialTransform
import SimpleITK as sitk
import numpy as np
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
import os
from random import random
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from LC_unet import LC_UNet


class Kits19(Dataset):
    def __init__(self, root_dir, transform=None):
        # self.root_dir = root_dir
        assert os.path.isdir(root_dir) == True, "root dir not exist"
        self.transform = transform
        # path=os.path.join(root_dir)
        self.images = []
        self.labels = []
        subdirs = [x[0] for x in os.walk(root_dir)]
        for subdir in subdirs:
            self.images.append(os.path.join(subdir, 'imaging.nii.gz'))
            self.labels.append(os.path.join(subdir, 'segmentation.nii.gz'))

    def __getitem__(self, index):
        image = sitk.ReadImage(self.images[index])
        image = sitk.GetArrayFromImage(image)
        label = sitk.ReadImage(self.labels[index])
        label = sitk.GetArrayFromImage(label)
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        sample = {'image': image, 'label': label}
        return sample

    def __len__(self):
        assert len(self.images) == len(self.labels)
        return len(self.images)


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        def random_rot_flip(image, label):
            k = np.random.randint(0, 4)
            image = np.rot90(image, k)
            label = np.rot90(label, k)
            axis = np.random.randint(0, 2)
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()
            return image, label

        def random_rotate(image, label):
            angle = np.random.randint(-20, 20)
            image = ndimage.rotate(image, angle, order=0, reshape=False)
            label = ndimage.rotate(label, angle, order=0, reshape=False)
            return image, label

        image, label = sample['image'], sample['label']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(
            image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        return sample


def argparse():
    def source_path(path):
        if os.path.isdir(path):
            return path
        else:
            raise argparse.ArgumentTypeError(f"output 2d slice directory:{path} is not a valid path")

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', help='batch size', default=2, type=int)
    parser.add_argument('-d', '--dataset', help='path to kits19 dataset', default='/media/research/dataset/kits19/data',
                        type=source_path)
    parser.add_argument('-p', '--patch_size', help='patch size', type=list, default=[256, 256])
    return parser.parse_args()


if __name__ == '__main__':
    args = argparse()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataset = Kits19(root_dir=args.dataset, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    model = LC_UNet.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train_loss = []
    train_epochs_loss = []
    for epoch in range(args.epochs):
        train_epoch_loss = []
        for idx, sample in enumerate(train_dataloader):
            image = sample['image'].to(device)
            label = sample['label'].to(device)
            outputs = model(image)
            outputs_soft = torch.softmax(outputs, dim=1)
            optimizer.zero_grad()
            loss = criterion(label, outputs_soft)
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
            train_loss.append(loss.item())
            if idx % (len(train_dataloader) // 2) == 0:
                print("epoch={}/{},{}/{}of train, loss={}".format(
                    epoch, args.epochs, idx, len(train_dataloader), loss.item()))
        train_epochs_loss.append(np.average(train_epoch_loss))
