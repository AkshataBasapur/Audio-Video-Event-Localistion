# 🎥🔊 Audio-Visual Event Localization (AVEL)

This project implements an **Audio-Visual Event Localization (AVEL)** system that detects and temporally localizes events in videos by jointly modeling **audio and visual modalities**.  
Unlike unimodal systems, AVEL leverages cross-modal learning, attention mechanisms, and temporal modeling to improve performance in noisy and occluded environments.

---

## 🚀 Features
- **Multimodal Fusion**: Combines audio (VGGish) and visual (ResNet-150) features.  
- **Attention Mechanisms**:  
  - Self-Attention for intra-modality temporal dependencies.  
  - Bimodal Cross-Attention for inter-modality feature alignment.  
- **Temporal Modeling**: Bidirectional LSTM (Bi-LSTM) captures sequential event dynamics.  
- **Event Localization**: Classifies and localizes events per video segment.  
- **High Accuracy**: Achieved **76.6% accuracy**, outperforming baseline models.  

---

## 📂 Architecture Overview
1. **Feature Extraction**  
   - Visual: ResNet-150 (2048-dim features)  
   - Audio: VGGish (128-dim features)  

2. **Attention Layers**  
   - Self-Attention for temporal focus within each modality.  
   - Bidirectional Cross-Modal Attention (Visual↔Audio).  

3. **Fusion & Temporal Modeling**  
   - Concatenated features → Bi-LSTM → Fully Connected (FC) layer.  

4. **Classification & Localization**  
   - Outputs per-second event labels across **28 event classes**.  

---

## 📊 Dataset
- **AVE Dataset**:  
  - 4,143 10-second videos.  
  - 28 event categories (e.g., dog barking, speech, vehicle sounds).  
  - Annotations include event start and end times.  
  - Train/Validation/Test split: 3339 / 402 / 402.  

---

## 📈 Results
| Model       | Accuracy (%) |
|-------------|--------------|
| Bi-AVAtt    | 72.53        |
| Bi-VAAtt    | 72.32        |
| **AVEL**    | **76.60**    |

- AVEL provides better **robustness under noise** and **event ambiguity** compared to unimodal/baseline systems.  

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Frameworks:** PyTorch / TensorFlow (depending on implementation)  
- **Models Used:**  
  - ResNet-150 (ImageNet pre-trained)  
  - VGGish (AudioSet pre-trained)  
  - Bi-LSTM  

---
