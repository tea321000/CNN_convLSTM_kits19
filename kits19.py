import os
from torch.utils.data import Dataset
import SimpleITK as sitk

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
            if os.path.isfile(os.path.join(subdir, 'imaging.nii.gz')):
                self.images.append(os.path.join(subdir, 'imaging.nii.gz'))
                self.labels.append(os.path.join(subdir, 'segmentation.nii.gz'))
        assert len(self.images) == len(self.labels)

    def __getitem__(self, index):
        image = sitk.ReadImage(self.images[index])
        image = sitk.GetArrayFromImage(image)
        label = sitk.ReadImage(self.labels[index])
        label = sitk.GetArrayFromImage(label)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.images)
