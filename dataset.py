import chainer


class CelebADataset(chainer.dataset.DatasetMixin):
    def __init__(self, paths, root):
        self.base = chainer.datasets.ImageDataset(paths, root)
        self.crop_size = (108, 108)

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        image = self.base[i]
        _, h, w = image.shape
        top = (h - self.crop_size[0]) // 2
        left = (w - self.crop_size[1]) // 2
        bottom = top + self.crop_size[0]
        right = left + self.crop_size[1]
        image = image[:, top:bottom, left:right]
        return image
