# 🚢 4Channel-YOLO-Maritime-Detection

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![YOLO](https://img.shields.io/badge/YOLO-v11-00FFFF)
![Rasterio](https://img.shields.io/badge/Geospatial-Rasterio-green)

A custom **4-channel YOLO pipeline** for multispectral Small Object Detection (SOD) in the maritime domain using Sentinel-2 satellite imagery. 

This repository demonstrates a complete end-to-end MLOps workflow, including PyTorch architecture surgery, data-centric weak supervision refinement, Monte Carlo Dropout for epistemic uncertainty quantification, and environmental noise ablation studies.

## 🧠 Core Technical Achievements
* **Custom Architecture Surgery:** Modified standard YOLO convolutional layers via PyTorch to natively accept 4-channel tensors (RGB + Near-Infrared) while preserving pre-trained weights via transfer learning.
* **Data-Centric AI:** Successfully transitioned from weak OSM (OpenStreetMap) labels to a refined dataset, jumping mAP@50 from 0.15 to highly competitive baseline metrics using only 50 manually curated training examples.
* **Epistemic Uncertainty Quantification:** Implemented stochastic forward passes (Monte Carlo Dropout) to generate spatial doubt heatmaps, successfully isolating structural anomalies (e.g., white roofs on land) from high-confidence maritime targets.
* **Robustness Ablation Study:** Mathematically validated the spatial value of the Near-Infrared (NIR) spectrum against standard RGB baselines under simulated atmospheric noise degradation.

---

## 📂 Repository Structure

```text
4Channel-YOLO-Maritime-Detection/
├── assets/                     # Visualizations, plots, and inference samples
├── configs/                    # YAML configuration files for model training
├── src/                        # Modular source code
│   ├── data/                   # Scripts for data preprocessing and label generation
│   ├── models/                 # Custom PyTorch surgery and training pipelines
│   └── evaluation/             # Inference, uncertainty heatmaps, and ablation scripts
├── .gitignore                  # Prevents uploading large TIFFs and model weights
├── requirements.txt            # Explicit environment dependencies
└── README.md
```

📊 Results & Scientific Validation
1. Training Convergence
The custom 4-channel model achieved stable convergence rapidly. Utilizing initialized weights (where the NIR channel was initialized using the mean of the RGB weights) prevented catastrophic forgetting during early epochs.

2. Ablation Study: Multispectral vs. RGB Robustness
To prove the value of the 4th (NIR) channel, an ablation study was conducted injecting incremental Gaussian noise (simulating atmospheric haze and sensor degradation) into the dataset.

Conclusion: The Custom 4-Channel model consistently outperforms the Baseline 3-Channel (RGB-only) model in clean and lightly degraded environments (noise levels 0 to 4). As noise increases beyond this threshold, both models experience expected signal collapse due to the fundamentally tiny pixel footprint of maritime vessels (<10 pixels), highlighting the critical limits of Small Object Detection.

3. Epistemic Uncertainty & Noise Filtration
Using Monte Carlo Dropout, stochastic forward passes were aggregated to visualize the model's spatial doubt.

Blue Regions: High epistemic doubt (The model investigated land-based infrastructure with similar spectral signatures but lacked consensus).

Deep Red Regions: High mathematical consensus (True maritime targets).

By adjusting the confidence threshold, we can actively filter out the "noisy" spatial doubt and isolate true positive detections. In a production environment, the remaining land-based noise would be handled via standard GeoJSON vector masking.

🚀 Quick Start & Usage
```
1. Clone and Install Dependencies:
  git clone [https://github.com/eraygencc/4Channel-YOLO-Maritime-Detection.git](https://github.com/YOUR_USERNAME/4Channel-YOLO-Maritime-Detection.git)
cd 4Channel-YOLO-Maritime-Detection
pip install -r requirements.txt
```
2. Train the Custom Model:

```
python src/models/train_4ch.py
```

3. Run the Ablation Study:

```
python src/evaluation/evaluate_noise.py
```

4. Generate Uncertainty Heatmaps:

```
python src/evaluation/mc_dropout_heatmap.py
```
Note: Due to GitHub file size limits, the data/ folder and .pt model weights are omitted. To run inference, supply your own 4-channel TIFF imagery.
