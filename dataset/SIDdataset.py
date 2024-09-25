import json

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from dataset.LLdataset import llDataset


class SIDdataset(llDataset):
    def __init__(
            self,
            image_size=(480, 720),
            dataset_type='train'
    ):
        super().__init__(
            image_size=image_size,
        )
        dataset_type_dict = {
            'train': '0',
            'val': '1',
            'test': '2'
        }
        self.dataset_type = dataset_type_dict[dataset_type]

        with open('meta_data.json', 'r') as f:
            meta_data = json.load(f)

        for data in meta_data:
            if data['type'] != self.dataset_type:
                continue
            # print(data)
            ll_image_path = data['ll_path']
            rgb_image_path = data['rgb_path']
            depth_image_path = data['depth_path']

            self.ll_image_paths.append(ll_image_path)
            self.rgb_image_paths.append(rgb_image_path)
            self.depth_image_paths.append(depth_image_path)

        self._get_len()


if __name__ == "__main__":
    dataset = llDataset()
    for i in range(10):
        ll, rgb, depth = dataset[i]
