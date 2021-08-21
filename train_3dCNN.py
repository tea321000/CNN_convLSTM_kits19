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
from torchvision.transforms import ToTensor

from threeDCNN import ThreeDCNN


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


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample['image'], sample['label']


        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 +
                                                      self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 +
                                                      self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


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
    parser.add_argument('-p', '--patch_size', help='patch size', type=list, default=[96, 96, 96])
    return parser.parse_args()


if __name__ == '__main__':
    args = argparse()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataset = Kits19(root_dir=args.dataset, transform=transforms.Compose([
        RandomRotFlip(),
        RandomCrop(args.patch_size),
        ToTensor(),
    ]))
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    model = ThreeDCNN.to(device)
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
