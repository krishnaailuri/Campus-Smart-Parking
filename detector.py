import torch
import torch.nn as nn
import torch.optim as optim
from model import ParkingSpaceModel, nms
from load_data import get_transforms
from PIL import Image

class ParkingLotDetector:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = ParkingSpaceModel().to(self.device)
        self.bbox_loss_weight = 1.0
        self.occupancy_loss_weight = 1.0

    def train_one_epoch(self, train_loader, optimizer, epoch):
        self.model.train()
        total_loss, bbox_losses, occupancy_losses = 0, 0, 0
        from tqdm import tqdm
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
        from tqdm import tqdm
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
        image = Image.open(image_path).convert('RGB')
        transform = get_transforms(train=False)
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        self.model.eval()
        outputs = self.model(image_tensor)[0]  # (5, H, W)
        bbox_map = outputs[:4, :, :]
        occupancy_map = torch.sigmoid(outputs[4, :, :])
        H, W = occupancy_map.shape
        boxes = []
        scores = []
        box_width = 30
        box_height = 60
        for y in range(H):
            for x in range(W):
                score = occupancy_map[y, x].item()
                if score >= conf_threshold:
                    center_x = (x + 0.5) * (640 / W)
                    center_y = (y + 0.5) * (640 / H)
                    x1 = center_x - box_width / 2
                    y1 = center_y - box_height / 2
                    x2 = center_x + box_width / 2
                    y2 = center_y + box_height / 2
                    boxes.append([x1, y1, x2, y2])
                    scores.append(score)
        if len(boxes) == 0:
            return {'boxes': [], 'occupancy': [], 'total_slots': 0,
                    'occupied_slots': 0, 'available_slots': 0}
        boxes_tensor = torch.tensor(boxes)
        scores_tensor = torch.tensor(scores)
        keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold)
        kept_boxes = boxes_tensor[keep_indices]
        flat_occupancy = occupancy_map.view(-1)
        occupancy = []
        for idx in keep_indices:
            occ_score = flat_occupancy[idx].item()
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
