# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ========== CONFIGURATION ==========
visual_attended_root = "/home/dgx1user3/Localization/visual_features_attended/"
audio_attended_root = "/home/dgx1user3/Localization/audio_attended"
fused_root = "/home/dgx1user3/Localization/fused_features"
label_file = "/home/dgx1user3/Localization/labels.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits = ["train", "val", "test"]
num_classes = 28
batch_size = 32
num_epochs = 100
learning_rate = 1e-4
early_stopping_patience = 5

# Create directories
for split in splits:
    os.makedirs(os.path.join(fused_root, split), exist_ok=True)

# Load labels
try:
    label_df = pd.read_csv(label_file)
    if not {'video_id', 'label'}.issubset(label_df.columns):
        raise ValueError("Label file must contain 'video_id' and 'label' columns")
    label_map = dict(zip(label_df["video_id"], label_df["label"]))
except FileNotFoundError:
    print(f"Error: {label_file} not found. Run convert_annotations.py first.")
    exit(1)
except Exception as e:
    print(f"Error reading {label_file}: {type(e).__name__}: {str(e)}")
    exit(1)

# Get video IDs
video_lists = {}
for split in splits:
    split_dir = os.path.join(visual_attended_root, split)
    if os.path.exists(split_dir):
        video_lists[split] = [f.replace('.npy', '') for f in os.listdir(split_dir) if f.endswith('.npy')]
    else:
        print(f"Warning: Directory {split_dir} does not exist.")
        video_lists[split] = []

# ========== STEP 1: SPLIT DATASET ==========
# (Commented out as dataset is already split)
# import random
# import shutil
# input_dir = "/home/dgx1user3/Localization/AVE/AVE/"
# output_base = "/home/dgx1user3/Localization/AVE/output"
# train_ratio = 0.7
# val_ratio = 0.15
# test_ratio = 0.15
# seed = 42
# random.seed(seed)
# all_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
# random.shuffle(all_files)
# total_files = len(all_files)
# train_end = int(total_files * train_ratio)
# val_end = train_end + int(total_files * val_ratio)
# train_files = all_files[:train_end]
# val_files = all_files[train_end:val_end]
# test_files = all_files[val_end:]
# for file_list, split in zip([train_files, val_files, test_files], splits):
#     for file in file_list:
#         src = os.path.join(input_dir, file)
#         dst = os.path.join(output_base, split, file)
#         shutil.copy2(src, dst)

# ========== STEP 2: VIDEO PREPROCESSING FUNCTIONS ==========
# (Commented out as videos are already preprocessed)
# import cv2
# def preprocess_video(video_path, output_dir, fps=16):
#     frames_dir = f"{output_dir}/frames"
#     audio_dir = f"{output_dir}/audio"
#     os.makedirs(frames_dir, exist_ok=True)
#     os.makedirs(audio_dir, exist_ok=True)
#     frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
#     if len(frame_files) == 160:
#         return
#     cap = cv2.VideoCapture(video_path)
#     video_fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_interval = int(video_fps / fps)
#     saved_id = 0
#     frame_id = 0
#     while cap.isOpened() and saved_id < 160:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if frame_id % 160 == 0:
#             frame_path = os.path.join(frames_dir, f"frame_{saved_id:04d}.jpg")
#             cv2.imwrite(frame_path, frame)
#             saved_id += 1
#         frame_id += 1
#     cap.release()

# ========== STEP 3: PREPROCESS DATA ==========
# (Commented out as preprocessing is done)
# preprocessed_root = "/home/dgx1user3/Localization/predata"
# for split in splits:
#     input_path = os.path.join(output_base, split)
#     output_path = os.path.join(preprocessed_root, split)
#     os.makedirs(output_path, exist_ok=True)
#     video_files = [f for f in os.listdir(input_path) if f.endswith('.mp4')]
#     for video_file in video_files:
#         video_path = os.path.join(input_path, video_file)
#         video_id = os.path.splitext(video_file)[0]
#         output_dir = os.path.join(output_path, video_id)
#         preprocess_video(video_path, output_dir, fps=16)

# ========== STEP 4: VISUAL FEATURE EXTRACTION ==========
# (Commented out as features are already extracted)
# from PIL import Image
# import torchvision.models as models
# import torchvision.transforms as transforms
# from torchvision.models.resnet import ResNet50_Weights
# resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
# resnet = torch.nn.Sequential(*list(resnet.children())[:1])
# resnet.eval().to(device)
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# def extract_visual_features(split):
#     input_dir = os.path.join(preprocessed_root, split)
#     output_dir = os.path.join(visual_features_root, split)
#     os.makedirs(output_dir, exist_ok=True)
#
#     video_ids = sorted(os.listdir(input_dir))
#     for video_id in video_ids:
#         frame_dir = os.path.join(input_dir, video_id, "frames")
#         if not os.path.isdir(frame_dir):
#             continue
#
#         frame_files = sorted([f for f in os.listdir(frame_dir) if f in f.endswith('.jpg')])
#         features = []
#
#         for frame_file in frame_files:
#             image = Image.open(os.path.join(frame_dir, frame_file)).convert('RGB')
#             tensor = transform(image).unsqueeze(0).to(device)
#             with torch.no_grad():
#                 feature = resnet(tensor).squeeze().cpu().numpy()
#             features.append(feature)
#
#         if features:
#             features_array = np.stack(features)
#             np.save(os.path.join(output_dir, f"{video_id}.npy"), features_array)

# ========== STEP 5: SIMPLE VISUAL_PROCESSOR ==========
# (Commented out as per user request)
# class SimpleVisualProcessor(nn.Module):
#     def __init__(self, input_dim=2048, output_dim=512):
#         super(SimpleVisualProcessor, self).__init__()
#         self.pool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Linear(input_dim, output_dim)
#
#     def forward(self, x):
#         # x: [batch, seq_len=160, dim=2048]
#         x = x.transpose(1, 2)  # [batch, 2048, 160]
#         x = x.contiguous()
#         x = self.pool(x).squeeze(-1)  # [batch, 2048]
#         x = self.fc(x)  # [batch, 512]
#         return x
#
# def process_visual_features():
#     model = SimpleVisualProcessor().to(device)
#     model.eval()
#
#     for split in splits:
#         input_dir = os.path.join(visual_features_root, split)
#         output_dir = os.path.join(fused_root, split)
#
#         for vid in tqdm(video_lists[split], desc=f"Processing {split}"):
#             input_path = os.path.join(input_dir, f"{vid}.npy")
#             output_path = os.path.join(output_dir, f"{vid}.npy")
#
#             if os.path.exists(output_path):
#                 continue
#
#             try:
#                 features = np.load(input_path)  # [160, 2048]
#                 if features.shape[0] != 160 or features.shape[1] != 2048:
#                     print(f"Skipping {vid}: Invalid shape {features.shape}, expected [160, 2048]")
#                     continue
#                 features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)  # [1, 160, 2048]
#
#                 with torch.no_grad():
#                     output = model(features)  # [1, 512]
#
#                 np.save(output_path, output.cpu().numpy())
#             except Exception as e:
#                 print(f"Error processing {vid}: {type(e).__name__}: {e}")
#                 continue

# ========== STEP 6: SELF-ATTENTION VISUAL PROCESSOR ==========
# (Retained but commented out for reference)
# class SelfAttention(nn.Module):
#     def __init__(self, feature_dim, num_heads=4):
#         super(SelfAttention, self).__init__()
#         self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
#         self.norm = nn.LayerNorm(feature_dim)
#     def forward(self, x):
#         x = x.unsqueeze(0)
#         attn_output, _ = self.attention(x, x, x)
#         out = self.norm(attn_output + x)
#         return out.squeeze(0)
# def apply_attention_to_features(input_root, output_root, model, feature_dim, split):
#     input_dir = os.path.join(input_root, split)
#     output_dir = os.path.join(output_root, split)
#     os.makedirs(output_dir, exist_ok=True)
#     files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])
#     for file in files:
#         input_path = os.path.join(input_dir, file)
#         output_path = os.path.join(output_dir, file)
#         features = np.load(input_path)
#         tensor = torch.tensor(features, dtype=torch.float32).to(device)
#         with torch.no_grad():
#             attended = model(tensor)
#             attended = attended.cpu().numpy()
#         np.save(output_path, attended)

# ========== STEP 7: BIMODAL ATTENTION ==========
class BiModalAttention(nn.Module):
    def __init__(self, primary_dim, aux_dim, output_dim=384):
        super(BiModalAttention, self).__init__()
        self.aux_proj = nn.Linear(aux_dim, primary_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim=primary_dim, num_heads=4, batch_first=True)
        self.primary_norm = nn.LayerNorm(primary_dim)
        self.aux_norm = nn.LayerNorm(primary_dim)
        self.fc = nn.Linear(primary_dim, output_dim)
    
    def forward(self, primary, aux):
        # primary: [batch, seq_len=160, primary_dim], aux: [batch, seq_len=160, aux_dim]
        primary = self.primary_norm(primary)  # [batch, 160, primary_dim]
        aux = self.aux_proj(aux)  # [batch, 160, aux_dim] ? [batch, 160, primary_dim]
        aux = self.aux_norm(aux)  # [batch, 160, primary_dim]
        
        # Cross-attention: primary attends to aux
        attn_output, _ = self.cross_attention(primary, aux, aux)  # [batch, 160, primary_dim]
        attn_output = self.primary_norm(attn_output + primary)  # [batch, 160, primary_dim]
        
        # Pool over sequence
        output = attn_output.mean(dim=1)  # [batch, primary_dim]
        output = self.fc(output)  # [batch, output_dim]
        return output

# ========== STEP 8: FUSE BIMODAL ATTENDED FEATURES ==========
def perform_bimodal_fusion():
    # Initialize two bimodal attention models
    model_audio = BiModalAttention(primary_dim=128, aux_dim=2048, output_dim=384).to(device)
    model_video = BiModalAttention(primary_dim=2048, aux_dim=128, output_dim=384).to(device)
    
    # Fusion layer to combine outputs
    fusion_fc = nn.Linear(384 * 2, 512).to(device)
    
    model_audio.eval()
    model_video.eval()
    fusion_fc.eval()
    
    for split in splits:
        v_dir = os.path.join(visual_attended_root, split)
        a_dir = os.path.join(audio_attended_root, split)
        f_dir = os.path.join(fused_root, split)
        os.makedirs(f_dir, exist_ok=True)
        
        for vid in tqdm(video_lists[split], desc=f"Fusing {split}"):
            v_path = os.path.join(v_dir, f"{vid}.npy")
            a_path = os.path.join(a_dir, f"{vid}.npy")
            out_path = os.path.join(f_dir, f"{vid}.npy")
            
            if os.path.exists(out_path):
                continue
            
            try:
                v_feat = np.load(v_path)  # [160, 2048]
                a_feat = np.load(a_path)  # [160, 128]
                
                # Validate shapes
                if v_feat.shape != (160, 2048):
                    print(f"Skipping {vid}: Invalid visual shape {v_feat.shape}, expected [160, 2048]")
                    continue
                if a_feat.shape != (160, 128):
                    print(f"Skipping {vid}: Invalid audio shape {a_feat.shape}, expected [160, 128]")
                    continue
                
                v_feat = torch.tensor(v_feat, dtype=torch.float32).unsqueeze(0).to(device)  # [1, 160, 2048]
                a_feat = torch.tensor(a_feat, dtype=torch.float32).unsqueeze(0).to(device)  # [1, 160, 128]
                
                with torch.no_grad():
                    # Audio-attended output
                    audio_out = model_audio(a_feat, v_feat)  # [1, 384]
                    # Video-attended output
                    video_out = model_video(v_feat, a_feat)  # [1, 384]
                    
                    # Concatenate outputs
                    concat_out = torch.cat([audio_out, video_out], dim=1)  # [1, 384*2=768]
                    
                    # Fused output
                    fused_out = fusion_fc(concat_out)  # [1, 512]
                    
                    # Validate fused output shape
                    if fused_out.shape != (1, 512):
                        print(f"Skipping {vid}: Invalid fused shape {fused_out.shape}, expected [1, 512]")
                        continue
                
                np.save(out_path, fused_out.cpu().numpy())
            except Exception as e:
                print(f"Error fusing {vid}: {type(e).__name__}: {str(e)}")
                continue

# ========== STEP 9: BILSTM CLASSIFIER ==========
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size=512, hidden_dim=256, num_layers=1, num_classes=28):
        super(BiLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers,
                            bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        # x: [batch, seq_len=1, input_size=512]
        lstm_out, _ = self.lstm(x)  # [batch, seq_len=1, hidden_dim*2]
        final = lstm_out[:, -1, :]  # [batch, hidden_dim*2]
        logits = self.fc(final)  # [batch, num_classes]
        return logits

# ========== STEP 10: DATASET AND TRAINING ==========
class FusedFeatureDataset(Dataset):
    def __init__(self, feature_dir, label_map, video_ids):
        self.feature_dir = feature_dir
        self.video_ids = video_ids
        self.label_map = label_map
    
    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        feature_path = os.path.join(self.feature_dir, f"{vid}.npy")
        features = np.load(feature_path).astype(np.float32)  # [512]
        # Shape for BiLSTM: [seq_len=1, input_size=512]
        features = torch.tensor(features).unsqueeze(0)  # [1, 512]
        try:
            label = self.label_map.get(vid)
            if label is None:
                raise KeyError(f"No label found for {vid}")
        except KeyError:
            print(f"Warning: No label found for video ID {vid}. Skipping.")
            return None
        return features, label

def train_model(model, train_loader, val_loader, device, num_epochs, patience):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    
    best_val_acc = 0.0
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            if batch is None:
                continue
            x, y = batch
            x, y = x.to(device), y.to(device)  # x: [batch, 1, 512]
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Compute train accuracy
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == y).sum().item()
            total_train += y.size(0)
        
        avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        train_acc = correct_train / total_train if total_train > 0 else 0.0
        
        # Validation accuracy
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}/{num_epochs}"):
                if batch is None:
                    continue
                x, y = batch
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = torch.argmax(logits, dim=1)
                correct_val += (preds == y).sum().item()
                total_val += y.size(0)
        
        val_acc = correct_val / total_val if total_val > 0 else 0.0
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_acc*100:.2f}%")
        print(f"Validation Accuracy: {val_acc*100:.2f}%")
        
        # Save best model and check for early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), "/home/dgx1user3/Localization/best_model.pth")
            print(f"Saved best model with validation accuracy: {val_acc*100:.2f}%")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{patience} epochs")
        
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            if batch is None:
                continue
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    test_acc = correct / total if total > 0 else 0.0
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    return test_acc

def predict_events(model, test_loader, device, video_ids):
    model.eval()
    predictions = []
    true_labels = []
    valid_video_ids = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch is None:
                continue
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(y.cpu().numpy())
            start_idx = batch_idx * test_loader.batch_size
            batch_vids = video_ids[start_idx:start_idx + len(preds)]
            valid_video_ids.extend(batch_vids)
    
    print("\nEvent Predictions:")
    print("Video ID | Predicted Label | True Label")
    print("-" * 40)
    for vid, pred, true in zip(valid_video_ids, predictions, true_labels):
        print(f"{vid} | {pred} | {true}")

# Run pipeline
print("Fusing bimodal features...")
perform_bimodal_fusion()

print("\nPreparing datasets...")
data_dirs = {s: os.path.join(fused_root, s) for s in splits}
for s in splits:
    try:
        video_lists[s] = [f.replace(".npy", "") for f in os.listdir(data_dirs[s]) if f.endswith('.npy')]
    except FileNotFoundError:
        print(f"Error: Directory {data_dirs[s]} not found. Ensure feature processing outputs are generated.")
        exit(1)

train_dataset = FusedFeatureDataset(data_dirs['train'], label_map, video_lists['train'])
val_dataset = FusedFeatureDataset(data_dirs['val'], label_map, video_lists['val'])
test_dataset = FusedFeatureDataset(data_dirs['test'], label_map, video_lists['test'])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

print("\nTraining model...")
model = BiLSTMClassifier(num_classes=num_classes)
train_model(model, train_loader, val_loader, device, num_epochs, early_stopping_patience)

print("\nEvaluating model...")
evaluate_model(model, test_loader, device)

print("\nPredicting events...")
predict_events(model, test_loader, device, video_lists['test'])