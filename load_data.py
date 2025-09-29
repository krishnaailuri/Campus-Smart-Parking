import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ParkingLotDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        self.images = self.annotations['images']
        self.image_to_anns = {}
        for ann in self.annotations['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_to_anns:
                self.image_to_anns[img_id] = []
            self.image_to_anns[img_id].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        anns = self.image_to_anns.get(img_id, [])
        boxes = []
        occupancy = []
        for ann in anns:
            bbox = ann['bbox']
            boxes.append([
                bbox[0],
                bbox[1],
                bbox[0] + bbox[2],
                bbox[1] + bbox[3]
            ])
            occupancy.append(ann['category_id'])
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            occupancy = torch.tensor(occupancy, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            occupancy = torch.zeros(0, dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return {
            'image': image,
            'boxes': boxes,
            'occupancy': occupancy,
            'image_id': img_id
        }
        
def collate_fn(batch):
    images, boxes, occupancy, image_ids = [], [], [], []
    for item in batch:
        images.append(item['image'])
        boxes.append(item['boxes'])
        occupancy.append(item['occupancy'])
        image_ids.append(item['image_id'])
    images = torch.stack(images)
    return {
        'images': images,
        'boxes': boxes,
        'occupancy': occupancy,
        'image_ids': image_ids
    }
    
def get_transforms(train=True):
    transforms_list = [
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]
    if train:
        transforms_list.insert(1, transforms.RandomHorizontalFlip(0.5))
        transforms_list.insert(1, transforms.ColorJitter(0.2, 0.2, 0.2))
    return transforms.Compose(transforms_list)
    
def create_data_loaders(data_path, batch_size=8):
    train_dataset = ParkingLotDataset(
        root_dir=os.path.join(data_path, 'train'),
        annotation_file=os.path.join(data_path, 'train/_annotations.coco.json'),
        transform=get_transforms(train=True)
    )
    valid_dataset = ParkingLotDataset(
        root_dir=os.path.join(data_path, 'valid'),
        annotation_file=os.path.join(data_path, 'valid/_annotations.coco.json'),
        transform=get_transforms(train=False)
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn
    )
    return train_loader, valid_loader
