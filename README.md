# NFL Big Data Bowl 2026: Player Trajectory Prediction

This repository contains a PyTorch implementation of a **Spatio-Temporal Graph Attention Network (ST-GAT)** designed to predict NFL player trajectories using Next Gen Stats tracking data.

## 📌 Project Overview
The goal of this project is to model player movements by capturing:
1.  **Spatial Interactions**: How players interact with teammates and opponents on the field (e.g., blocking, coverage).
2.  **Temporal Dynamics**: How movement evolves over time (velocity, acceleration profiles).

We treat the 22 players on the field as nodes in a fully connected graph, where edge weights (attention scores) represent the importance of one player's movement to another.

## 🧠 Model Architecture
The core model (`STGATModel`) combines Graph Neural Networks with Sequence models:

1.  **Input Layer**:
    - Features: X, Y, Speed, Acceleration, Orientation, Direction, Player Weight, etc.
    - Categorical Embeddings: Team, Position, Role.
2.  **Spatial Block (GATv2)**:
    - Uses **Graph Attention Networks (GATv2Conv)** to perform message passing between all 22 players in each frame.
    - Allows the model to dynamically "attend" to relevant players (e.g., a Linebacker accurately tracking a Running Back) while ignoring irrelevant ones.
3.  **Temporal Block (LSTM)**:
    - Processes the sequence of spatial embeddings over time.
    - Captures the trajectory history (past 10 frames) to predict future states.
4.  **Decoder**:
    - Predicts the (X, Y) coordinates for future time steps.

## 📂 File Structure
```
gat-lstm/
├── nfl_train.py           # Main training script (Data Loading + Model + Train Loop)
├── submission_script.py   # Inference script to generate predictions and metrics
├── check_gpu.py           # Utility to verify CUDA/GPU availability
├── requirements.txt       # Python dependencies
└── README.md              # This documentation
```

## 🚀 Installation

1.  **Clone the repository**:
    ```bash
    git clone <your-repo-url>
    cd gat-lstm
    ```

2.  **Install Dependencies**:
    Recommended to use a Virtual Environment (Python 3.10+).
    ```bash
    pip install -r requirements.txt
    ```
    *Note: For GPU acceleration, ensure you install the correct PyTorch version for your CUDA driver from [pytorch.org](https://pytorch.org).*

## 🏋️ usage

### 1. Training
To train the model from scratch on the full dataset:
```bash
python nfl_train.py
```
- **Configuration**: Edit the `CONFIG` dictionary in `nfl_train.py` to change batch size, epochs, or data paths.
- **Output**: The best model will be saved as `st_gat_model.pth`.

### 2. Inference / Evaluation
To generate predictions and calculate accuracy metrics (MSE, RMSE, Accuracy):
```bash
python submission_script.py
```
- This script loads the trained `st_gat_model.pth`.
- It calculates metrics on the validation set (Week 18).
- Generates a `submission.csv` file with predicted coordinates.

## 📊 Data Preparation
This solution expects the standard NFL Big Data Bowl dataset structure:
- `train/`: Directory containing specific week files (`input_2023_wXX.csv` and `output_2023_wXX.csv`).
- Data is processed using the `NFLBigDataDataset` class, which handles:
    - **Scaling**: StandardScalar for numerical features (x, y, s, a, etc.).
    - **Encoding**: LabelEncoding for categorical features.
    - **Imputation**: Handling missing values (e.g., player weight).

## 📈 Performance
The model performance is evaluated using:
- **RMSE (Root Mean Squared Error)**: Average deviation in yards.
- **Accuracy (< 1yd, < 2yd)**: Percentage of predictions within a specific distance threshold.

## 🛠 Tech Stack
- **Python 3.x**
- **PyTorch** (Deep Learning Framework)
- **PyTorch Geometric** (Graph Neural Networks)
- **Pandas / NumPy** (Data Manipulation)
- **Scikit-Learn** (Preprocessing)
