from load_data import create_data_loaders
from detector import ParkingLotDetector
import os
import torch

def train_model(train_loader, valid_loader, num_epochs=10):
    detector = ParkingLotDetector()
    optimizer = torch.optim.Adam(detector.model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
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


if __name__ == "__main__":
    path = "/Users/admin/Desktop/GIT/Campus_Smart_Parking/pklot-dataset"
    train_loader, valid_loader = create_data_loaders(path)
    print(f"Training dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(valid_loader.dataset)}")

    detector = train_model(train_loader, valid_loader)

    test_image_path = os.path.join(path, 'test/sample.jpg')
    if os.path.exists(test_image_path):
        results = detector.detect_image(test_image_path)
        print("\nTest Results:")
        print(f"Total Slots: {results['total_slots']}")
        print(f"Occupied Slots: {results['occupied_slots']}")
        print(f"Available Slots: {results['available_slots']}")
