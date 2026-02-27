from torch.utils.data import Dataset
from PIL import Image
import json
import os

class SODA10MDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        ann_path = os.path.join(root, "annotations", f"{split}.json")
        with open(ann_path, "r") as f:
            coco = json.load(f)

        # Build image id -> file name map
        self.images = {img["id"]: img for img in coco["images"]}

        # Build annotations per image
        self.ann_by_image = {}
        for ann in coco["annotations"]:
            iid = ann["image_id"]
            self.ann_by_image.setdefault(iid, []).append(ann)

        self.image_ids = list(self.ann_by_image.keys())

        # Map category ids to class indices
        self.cat_to_idx = {cat["id"]: i for i, cat in enumerate(coco["categories"])}

        self.targets = [
            self.cat_to_idx[self.ann_by_image[iid][0]["category_id"]]
            for iid in self.image_ids
        ]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        iid = self.image_ids[idx]
        img_info = self.images[iid]

        img_path = os.path.join(self.root, self.split, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Use the first annotation's category as the label (for classification)
        # For detection, you'd return all boxes/labels
        label = self.targets[idx] 

        return image, label