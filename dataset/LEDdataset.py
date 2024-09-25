from dataset.LLdataset import llDataset
from torchvision import transforms
from PIL import Image

from utils.data_preprocess import get_center_mask


class LEDdataset(llDataset):
    def __init__(
            self,
            image_size=(480, 720),
            dataset_type='train'
    ):
        super().__init__(image_size)
        image_size = (720, 480)
        self.image_size = (720, 480)
        self.dataset_type = dataset_type
        self.valid_p_depth_list = []
        self.read_valid_p_depth_list()
        if dataset_type == 'train':
            self.valid_p_depth_list = [path for path in self.valid_p_depth_list if 'train' in path]
        elif dataset_type == 'val':
            self.valid_p_depth_list = [path for path in self.valid_p_depth_list if 'val' in path]

        self.valid_p_depth_list = sorted(self.valid_p_depth_list)
        self.depth_image_paths = [path.replace('p_depth', 'depth') for path in self.valid_p_depth_list]
        self.ll_image_paths = [path.replace('p_depth', 'input') for path in self.valid_p_depth_list]
        self.rgb_image_paths = [path.replace('p_depth', 'target') for path in self.valid_p_depth_list]
        self._get_len()

        self.ll_image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.depth_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __getitem__(self, idx):
        # If idx is a list or a slice, return a list of items
        if isinstance(idx, list):
            return [self.__getitem__(i) for i in idx]
        elif isinstance(idx, slice):
            return [self.__getitem__(i) for i in self.idxs[idx]]

        # Low-light image, image , depth
        ll_image_path = self.ll_image_paths[idx]
        image_path = self.rgb_image_paths[idx]
        depth_path = self.depth_image_paths[idx]

        # Load low-light image
        ll_image = Image.open(ll_image_path).convert('RGB')
        image = Image.open(image_path).convert('RGB')
        depth = Image.open(depth_path).convert('L')
        # print(f"ll shape: {ll_image.size}")
        # print(f"rgb shape: {image.size}")
        # print(f"depth shape: {depth.size}")

        ll_image = self.ll_image_transform(ll_image).clamp(-1, 1).permute(0, 2, 1)
        image = self.image_transform(image).clamp(-1, 1).permute(0, 2, 1)
        depth = self.depth_transform(depth).clamp(-1, 1).permute(0, 2, 1)
        mask = (depth > -0.98) & (depth < 0.98)
        mask = mask & get_center_mask(mask, 0.1)
        # print(f"ll shape: {ll_image.shape}")
        # print(f"rgb shape: {image.shape}")
        # print(f"depth shape: {depth.shape}")
        return ll_image, image, depth, mask

    def read_valid_p_depth_list(self):
        with open("/dataset/vfayezzhang/test/DAv2/master/valid_path.txt", "r") as f:
            self.valid_p_depth_list = [line.strip() for line in f.readlines()]


if __name__ == "__main__":
    dataset = LEDdataset()
    print(f"dataset length: {len(dataset)}")
    for i in range(10):
        ll, rgb, depth = dataset[i]
        print(ll.shape, rgb.shape, depth.shape)
