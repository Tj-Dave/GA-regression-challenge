
---

# Blindsweep GA Prediction

![Python](https://img.shields.io/badge/Python-3.8%20|%203.9%20|%203.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)
![conda](https://img.shields.io/badge/conda-environment-44a833)
![nibabel](https://img.shields.io/badge/nibabel-5.2+-purple)
![numpy](https://img.shields.io/badge/numpy-1.24+-orange)
![pandas](https://img.shields.io/badge/pandas-2.0+-blueviolet)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

---

A clean and modular deep learning template to **predict Gestational Age (GA)** from **Blindsweep ultrasound** datasets using PyTorch.
Built for reproducibility, simplicity, and extensibility. 🚀🧠

---

## 📌 Introduction

### Why you might want to use it:

✅ **Ready-to-use template**
Includes datasets, models, training loops, and inference pipelines for GA prediction.

✅ **Attention-based video aggregation**
Uses a ResNet backbone and weighted average attention to handle multiple frames per sweep.

✅ **Educational and reproducible**
Thoroughly commented code and modular dataset/model classes make it easy to learn and extend.

---

### Why you might not want to use it:

❌ Not optimized for very large datasets (can be adapted).

❌ GPU is recommended; CPU training will be slow.

---

## 🛠 Features

* Handles **single and multi-sweep datasets**.
* Uses **ResNet18 backbone** with optional fine-tuning.
* Weighted average attention for frame aggregation.
* Training and validation with MAE loss.
* Saves best model automatically.
* **TensorBoard integration** for visualization.

---

## ⚙️ Environment Setup

### 1. Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
echo 'eval "$($HOME/miniconda3/bin/conda shell.bash hook)"' >> ~/.bashrc
source ~/.bashrc
```

### 2. Create & activate environment

```bash
conda create -n ga-us python=3.10
conda activate ga-us
```

### 3. Install dependencies

```bash
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install notebook pandas tensorboard
conda install -c conda-forge nibabel
pip install tqdm
```

### 4. Create project directories

```bash
mkdir checkpoints logs
```

---

## 📂 Dataset

* **Train dataset**: single sweep per sample.
* **Validation/Test dataset**: multiple sweeps per sample.
* CSV files should include:

```
study_id, ga, path_nifti_1, path_nifti_2, ...
```

---

## 📂 Folder Structure

```plaintext
.
├── .gitignore
├── check.ipynb
├── data.py
├── evaluate_metrics.py
├── infer.py
├── model.py
├── requirements.txt
├── train.py
└── checkpoints/
    └── best_model.pth
└── logs/
    ├── events.out.tfevents...
```

* `.gitignore`: Git ignore file for unnecessary files.
* `check.ipynb`: Jupyter notebook for exploratory analysis or debugging.
* `data.py`: Dataset loading and preprocessing logic.
* `evaluate_metrics.py`: Code to evaluate model performance.
* `infer.py`: Inference logic for prediction.
* `model.py`: Contains model architecture (ResNet backbone).
* `requirements.txt`: List of required Python packages.
* `train.py`: Training loop and validation logic.
* `checkpoints/`: Directory to store saved model checkpoints.
* `logs/`: Directory to store TensorBoard logs.

---

## 🏗 Model Architecture

* **Backbone:** ResNet18 (pretrained, optionally frozen).
* **Attention:** WeightedAverageAttention for frame aggregation.
* **Output:** Linear layer predicting GA.

Improve the model as required.

---

## 🚀 Training & Validation

Run training using:

```python
from train import train_and_validate

train_csv = "path/to/train.csv"
val_csv = "path/to/val.csv"
train_and_validate(train_csv, val_csv, epochs=100, batch_size=8, n_sweeps_val=8, save_path='checkpoints/best_model.pth')
```

* Uses **MAE (L1) loss**. The code can be adapted to use other loss functions.
* Saves **best model** automatically to `checkpoints/best_model.pth`.

### TensorBoard Integration

You can visualize the training process with TensorBoard. To log training metrics and visualize them:

1. **Start TensorBoard on the server**:

   Run the following command on the server where your model is training. This will start the TensorBoard service:

   ```bash
   tensorboard --logdir=logs --port=6000
   ```
   
   * `--port=6000` specifies the port to use (default is 6006).

2. **Map the server port to your local machine**:

   If you're connecting to the server remotely via SSH, you'll need to forward the TensorBoard port so you can access it locally. In your terminal (on your local machine), run:

   ```bash
   ssh -L -p 5555 6000:localhost:6000 user@server_ip
   ```

   Replace `user@server_ip` with your actual username and server IP. This command forwards the server's port 6006 (where TensorBoard is running) to your local machine's port 6006.

3. **Open TensorBoard in your browser**:

   Once the SSH tunnel is established, open your browser and navigate to `http://localhost:6006` to view TensorBoard and monitor metrics like loss, accuracy, etc.

---

## 🧪 Inference

To predict GA from test data:

```python
from infer import predict_ga

model_path = "checkpoints/best_model.pth"
test_csv = "path/to/test.csv"
predictions = predict_ga(model_path, test_csv)
```

### Example Inference CSV

Here’s a sample of how your inference CSV should look:

```
study_id, site, predicted_ga
KA-PC-002-1, Kenya, 180
NL-PC-087-1, Nepal, 157
PN-PC-090-1, Pakiastan, 223
```

---

## 📖 References

* [PyTorch](https://pytorch.org/)
* [TorchVision](https://pytorch.org/vision/stable/index.html)
* [NiBabel](https://nipy.org/nibabel/) for NIfTI image handling

---

## 📝 License

MIT License.

---
