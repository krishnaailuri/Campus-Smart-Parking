# ======================= Imports =======================
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


# ======================= Dataset =======================
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
            bbox = ann['bbox']  # [x, y, width, height]
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


# ======================= Model =======================
class ParkingSpaceModel(nn.Module):
    def __init__(self):
        super(ParkingSpaceModel, self).__init__()
        self.backbone = nn.Sequential(
            self._make_conv_block(3, 64),
            self._make_conv_block(64, 128),
            self._make_conv_block(128, 256),
            self._make_conv_block(256, 512),
        )
        self.detection_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 5, kernel_size=1)  # 4 bbox coords, 1 occupancy
        )

    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.detection_head(features)
        return output  # (B, 5, H, W)


# =================== Utility: Non-Maximum Suppression ===================
def nms(boxes, scores, iou_threshold=0.5):
    """
    Performs Non-Maximum Suppression on bounding boxes.
    boxes: (N,4) array of boxes [x1,y1,x2,y2]
    scores: (N,) array of scores
    """
    if len(boxes) == 0:
        return []

    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]  # sort by scores descending

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


# ======================= Detector & Training Utils =======================
class ParkingLotDetector:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = ParkingSpaceModel().to(self.device)
        self.bbox_loss_weight = 1.0
        self.occupancy_loss_weight = 1.0

    def train_one_epoch(self, train_loader, optimizer, epoch):
        self.model.train()
        total_loss, bbox_losses, occupancy_losses = 0, 0, 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(self.device)
            boxes = [b.to(self.device) for b in batch['boxes']]
            occupancy = [o.to(self.device) for o in batch['occupancy']]
            optimizer.zero_grad()
            outputs = self.model(images)
            batch_size = images.size(0)
            batch_loss, batch_bbox_loss, batch_occupancy_loss = 0, 0, 0

            for i in range(batch_size):
                pred_boxes = outputs[i, :4].view(-1, 4)
                pred_occupancy = outputs[i, 4].view(-1)
                if len(boxes[i]) > 0:
                    bbox_loss = nn.MSELoss()(pred_boxes[:len(boxes[i])], boxes[i])
                    batch_bbox_loss += bbox_loss
                    occupancy_loss = nn.BCEWithLogitsLoss()(
                        pred_occupancy[:len(occupancy[i])],
                        occupancy[i].float()
                    )
                    batch_occupancy_loss += occupancy_loss
                    img_loss = (self.bbox_loss_weight * bbox_loss +
                                self.occupancy_loss_weight * occupancy_loss)
                    batch_loss += img_loss
            if batch_size > 0:
                batch_loss /= batch_size
                batch_bbox_loss /= batch_size
                batch_occupancy_loss /= batch_size
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
            bbox_losses += batch_bbox_loss.item()
            occupancy_losses += batch_occupancy_loss.item()
            pbar.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.4f}',
                'bbox_loss': f'{bbox_losses/(batch_idx+1):.4f}',
                'occ_loss': f'{occupancy_losses/(batch_idx+1):.4f}'
            })
        return total_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(self, valid_loader):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_predictions = 0
        for batch in tqdm(valid_loader, desc='Evaluating'):
            images = batch['images'].to(self.device)
            boxes = [b.to(self.device) for b in batch['boxes']]
            occupancy = [o.to(self.device) for o in batch['occupancy']]
            outputs = self.model(images)
            batch_size = images.size(0)
            for i in range(batch_size):
                pred_boxes = outputs[i, :4].view(-1, 4)
                pred_occupancy = outputs[i, 4].view(-1)
                if len(boxes[i]) > 0:
                    bbox_loss = nn.MSELoss()(pred_boxes[:len(boxes[i])], boxes[i])
                    occupancy_loss = nn.BCEWithLogitsLoss()(
                        pred_occupancy[:len(occupancy[i])],
                        occupancy[i].float()
                    )
                    loss = (self.bbox_loss_weight * bbox_loss +
                            self.occupancy_loss_weight * occupancy_loss)
                    total_loss += loss.item()
                    pred_labels = (torch.sigmoid(pred_occupancy[:len(occupancy[i])]) > 0.5)
                    correct = (pred_labels == occupancy[i].bool()).sum().item()
                    total_correct += correct
                    total_predictions += len(occupancy[i])
        avg_loss = total_loss / len(valid_loader)
        accuracy = total_correct / total_predictions if total_predictions > 0 else 0
        return avg_loss, accuracy

    @torch.no_grad()
    def detect_image(self, image_path, conf_threshold=0.9, iou_threshold=0.4):
        """
        Detect parking spaces in a single image with post-processing.
        Returns boxes and occupancy predictions ready for visualization.
        """
        image = Image.open(image_path).convert('RGB')
        transform = get_transforms(train=False)
        image_tensor = transform(image).unsqueeze(0).to(self.device)

        self.model.eval()
        outputs = self.model(image_tensor)[0]  # (5, H, W)

        # Separate bbox coords and occupancy logits
        bbox_map = outputs[:4, :, :]  # shape (4, H, W)
        occupancy_map = torch.sigmoid(outputs[4, :, :])  # shape (H, W)

        H, W = occupancy_map.shape
        # Prepare candidate boxes from feature map grid locations
        boxes = []
        scores = []

        # Define a fixed box size relative to the resized image (640x640)
        box_width = 30
        box_height = 60

        for y in range(H):
            for x in range(W):
                score = occupancy_map[y, x].item()
                if score >= conf_threshold:
                    # Use model bbox coords: predicted offsets or absolute coords? Assuming offsets here
                    # For demonstration, create bounding boxes centered at grid cell with fixed size
                    center_x = (x + 0.5) * (640 / W)
                    center_y = (y + 0.5) * (640 / H)
                    x1 = center_x - box_width / 2
                    y1 = center_y - box_height / 2
                    x2 = center_x + box_width / 2
                    y2 = center_y + box_height / 2
                    boxes.append([x1, y1, x2, y2])
                    scores.append(score)

        if len(boxes) == 0:
            return {
                'boxes': [],
                'occupancy': [],
                'total_slots': 0,
                'occupied_slots': 0,
                'available_slots': 0
            }

        boxes_tensor = torch.tensor(boxes)
        scores_tensor = torch.tensor(scores)

        # NMS to remove overlapping boxes
        keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold)

        kept_boxes = boxes_tensor[keep_indices]
        kept_scores = scores_tensor[keep_indices]

        # occupancy_map is shape (H, W), reshape to (H*W,) to index by flattened indices
        flat_occupancy = occupancy_map.view(-1)

        # After NMS
        occupancy = []
        print("Occupancy scores of detected boxes:")
        for idx in keep_indices:
            occ_score = flat_occupancy[idx].item()
            print(occ_score)
            occupancy.append(1 if occ_score >= conf_threshold else 0)


        total_slots = len(kept_boxes)
        occupied_slots = sum(occupancy)
        available_slots = total_slots - occupied_slots

        return {
            'boxes': kept_boxes.cpu().numpy(),
            'occupancy': occupancy,
            'total_slots': total_slots,
            'occupied_slots': occupied_slots,
            'available_slots': available_slots
        }

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))


def train_model(train_loader, valid_loader, num_epochs=10):
    detector = ParkingLotDetector()
    optimizer = optim.Adam(detector.model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    best_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = detector.train_one_epoch(train_loader, optimizer, epoch)
        val_loss, val_accuracy = detector.evaluate(valid_loader)
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            detector.save_model('best_model.pth')
    return detector


# ======================= Usage =======================

if __name__ == "__main__":
    path = "/Users/admin/Desktop/GIT/Campus_Smart_Parking/pklot-dataset"
    train_loader, valid_loader = create_data_loaders(path)
    print(f"Training dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(valid_loader.dataset)}")

    for batch in train_loader:
        print("Batch size:", len(batch['images']))
        print("Image shape:", batch['images'].shape)
        print("Number of boxes in first image:", len(batch['boxes'][0]))
        print("Number of occupancy labels in first image:", len(batch['occupancy'][0]))
        break

    detector = train_model(train_loader, valid_loader)

    test_image_path = os.path.join(path, 'test/sample.jpg')
    if os.path.exists(test_image_path):
        results = detector.detect_image(test_image_path)
        print("\nTest Results:")
        print(f"Total Slots: {results['total_slots']}")
        print(f"Occupied Slots: {results['occupied_slots']}")
        print(f"Available Slots: {results['available_slots']}")
