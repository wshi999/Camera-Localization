import os
import torch.utils.data as data
from torchvision import transforms as T
from PIL import Image
import numpy as np
import torch


class DataSource(data.Dataset):
    def __init__(self, root, resize=256, crop_size=224, train=True):
        self.root = os.path.expanduser(root)
        self.resize = resize
        self.crop_size = crop_size
        self.train = train

        self.image_poses = []
        self.images_path = []

        self._get_data()

        # TODO: Define preprocessing
        self.pre_resize = T.Resize(resize)
        self.centercrop = T.CenterCrop(crop_size)
        self.randomcrop = T.RandomCrop(crop_size)
        self.normalize = T.Normalize(0.5, 0.5)
        self.totensor = T.ToTensor()
        # Load mean image
        self.mean_image_path = os.path.join(self.root, "mean_image.npy")
        if os.path.exists(self.mean_image_path):
            self.mean_image = np.load(self.mean_image_path)
            print("Mean image loaded!")
        else:
            self.mean_image = self.generate_mean_image()

    def _get_data(self):

        if self.train:
            txt_file = self.root + "dataset_train.txt"
        else:
            txt_file = self.root + "dataset_test.txt"

        with open(txt_file, "r") as f:
            next(f)  # skip the 3 header lines
            next(f)
            next(f)
            for line in f:
                fname, p0, p1, p2, p3, p4, p5, p6 = line.split()
                p0 = float(p0)
                p1 = float(p1)
                p2 = float(p2)
                p3 = float(p3)
                p4 = float(p4)
                p5 = float(p5)
                p6 = float(p6)
                self.image_poses.append((p0, p1, p2, p3, p4, p5, p6))
                self.images_path.append(self.root + fname)

    def generate_mean_image(self):
        print("Computing mean image:")

        # TODO: Compute mean image

        # Initialize mean_image
        mean_image = np.zeros((256, 455, 3), dtype=np.float)
        # Iterate over all training images
        # Resize, Compute mean, etc...
        for path in self.images_path:
            img = Image.open(path)
            img = self.pre_resize(img)
            np_img = np.array(img, dtype=np.float)
            mean_image = mean_image + np_img
        mean_image = mean_image / len(self.images_path)
        # Store mean image
        with open(self.mean_image_path, "wb") as f:
            np.save(f, mean_image)

        print("Mean image computed!")

        return mean_image

    def __getitem__(self, index):
        """
        return the data of one image
        """
        img_path = self.images_path[index]
        img_pose = self.image_poses[index]

        data = Image.open(img_path)

        # TODO: Perform preprocessing
        data = self.pre_resize(data)
        data = np.array(data, dtype=np.float)
        data = data - self.mean_image
        data = self.totensor(data)
        if self.train:
            data = self.randomcrop(data)
        else:
            data = self.centercrop(data)
        data = self.normalize(data)
        data.type(torch.float32)

        return data, img_pose

    def __len__(self):
        return len(self.images_path)
