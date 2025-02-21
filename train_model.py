import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import shutil
from tqdm import tqdm
import gc

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data augmentation and normalization for training
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def organize_rice_dataset():
    rice_path = "./Rice"
    if os.path.exists(os.path.join(rice_path, "validation")) and not os.listdir(os.path.join(rice_path, "val")):
        # Copy validation data to val directory
        validation_path = os.path.join(rice_path, "validation")
        val_path = os.path.join(rice_path, "val")
        if os.path.exists(validation_path):
            # Copy all contents from validation to val
            shutil.copytree(validation_path, val_path, dirs_exist_ok=True)
            print("Organized Rice validation data")

def load_datasets(data_dir):
    datasets_dict = {}
    
    # First organize Rice dataset
    organize_rice_dataset()
    
    for vegetable in ['Corn', 'Potato', 'Rice', 'Tomato']:
        veg_path = os.path.join(data_dir, vegetable)
        if not os.path.exists(veg_path):
            print(f"Skipping {vegetable} - directory not found")
            continue
            
        # Handle different directory structures
        train_dir = None
        val_dir = None
        
        # Check possible training directory names
        train_dirs = ['Train', 'train']
        val_dirs = ['Valid', 'val', 'validation']
        
        for d in train_dirs:
            if os.path.exists(os.path.join(veg_path, d)):
                train_dir = os.path.join(veg_path, d)
                break
                
        for d in val_dirs:
            if os.path.exists(os.path.join(veg_path, d)):
                val_dir = os.path.join(veg_path, d)
                break
        
        # For Corn, if no train/val directories with data, use disease class directories
        if vegetable == 'Corn' and (not train_dir or not os.listdir(train_dir)):
            disease_dirs = [d for d in os.listdir(veg_path) if os.path.isdir(os.path.join(veg_path, d)) and d.startswith('Corn_')]
            if disease_dirs:
                train_dir = os.path.join(veg_path, 'train')
                val_dir = os.path.join(veg_path, 'val')
                os.makedirs(train_dir, exist_ok=True)
                os.makedirs(val_dir, exist_ok=True)
                
                if not os.listdir(train_dir):
                    for disease in disease_dirs:
                        disease_path = os.path.join(veg_path, disease)
                        disease_name = disease.replace('Corn_', '')
                        train_disease_dir = os.path.join(train_dir, disease_name)
                        val_disease_dir = os.path.join(val_dir, disease_name)
                        
                        os.makedirs(train_disease_dir, exist_ok=True)
                        os.makedirs(val_disease_dir, exist_ok=True)
                        
                        files = os.listdir(disease_path)
                        split_idx = int(len(files) * 0.8)
                        
                        for i, file in enumerate(files):
                            src = os.path.join(disease_path, file)
                            if i < split_idx:
                                dst = os.path.join(train_disease_dir, file)
                            else:
                                dst = os.path.join(val_disease_dir, file)
                            shutil.copy2(src, dst)
        
        if train_dir and val_dir and os.path.exists(train_dir) and os.path.exists(val_dir):
            try:
                image_datasets = {
                    'train': datasets.ImageFolder(train_dir, data_transforms['train']),
                    'val': datasets.ImageFolder(val_dir, data_transforms['val'])
                }
                datasets_dict[vegetable] = image_datasets
                print(f"Successfully loaded {vegetable} dataset")
                print(f"Number of training images: {len(image_datasets['train'])}")
                print(f"Number of validation images: {len(image_datasets['val'])}")
                print(f"Classes: {image_datasets['train'].classes}")
            except Exception as e:
                print(f"Error loading {vegetable} dataset: {str(e)}")
                continue
        else:
            print(f"Skipping {vegetable} - missing train or validation directory")
            
    return datasets_dict

def create_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for param in list(model.parameters())[:-20]:
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_model(model, dataloaders, criterion, optimizer, num_epochs=10):
    model = model.to(device)
    best_acc = 0.0
    best_model_wts = None
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in tqdm(dataloaders[phase], desc=f'{phase} batches'):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                try:
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                except RuntimeError as e:
                    print(f"Error during batch processing: {str(e)}")
                    continue
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save best validation model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict().copy()
        
        gc.collect()
        print()
    
    # Load best model weights
    if best_model_wts:
        model.load_state_dict(best_model_wts)
    return model

def main():
    data_dir = "."
    batch_size = 16
    num_epochs = 10
    
    print("Loading datasets...")
    datasets_dict = load_datasets(data_dir)
    
    for vegetable, image_datasets in datasets_dict.items():
        print(f"\nTraining model for {vegetable}")
        
        dataloaders = {
            x: DataLoader(
                image_datasets[x], 
                batch_size=batch_size,
                shuffle=True, 
                num_workers=0
            )
            for x in ['train', 'val']
        }
        
        num_classes = len(image_datasets['train'].classes)
        print(f"Number of classes: {num_classes}")
        print("Class names:", image_datasets['train'].classes)
        
        model = create_model(num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        try:
            print(f"\nStarting training for {vegetable}...")
            model = train_model(model, dataloaders, criterion, optimizer, num_epochs)
            print(f"\nSaving model for {vegetable}...")
            torch.save(model.state_dict(), f'{vegetable}_disease_model.pth')
            print(f"Model saved as {vegetable}_disease_model.pth")
        except Exception as e:
            print(f"Error training {vegetable} model: {str(e)}")
            continue
        
        gc.collect()

if __name__ == "__main__":
    main()
