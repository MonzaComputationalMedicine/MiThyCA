# 🧠 MiThyCA: a democratic computational pathology pipeline for the identification of Microfoci of Thyroid Carcinoma with AI in whole-slide histological images

MiThyCA is a powerful deep learning tool that detects neoplastic and papillary thyroid carcinoma (PTC-only) regions on
whole-slide images (WSIs) in **real-time**, on any hardware, using **QuPath** for visualization. 🎯

---

## 🚀 Live Inference with QuPath

With MiThyCA, you can analyze a slide directly inside **QuPath**, and get predictions overlaid right onto the image. 📊

Here’s how to do it step by step:

### 🛠️ 1. Start the QuPath Server

MiThyCA connects to QuPath via a local server. To start that server:

**Option 1: Manual Script Execution**

1. Open QuPath.
2. Go to `Automate → Script Editor`.
3. Copy and paste the contents of `start_server.groovy`.
4. Run the script to start the server.

**Option 2: Save Script for Easy Access**

1. Open QuPath.
2. Go to `Automate → Shared Scripts → Open Scripts Directory`.
3. Copy `start_server.groovy` into this folder.
4. Then you’ll be able to start it easily from:
   `Automate → Shared Scripts → start server`.

📡 Once this server is running, you're ready to launch MiThyCA’s inference!

---

### 🧪 2. Run Inference

First, make sure to create an environment with the necessary packages.
With pip:
```bash
pip3 install torch torchvision torchaudio timm geojson py4j numpy openslide-python openslide-bin scikit-image Pillow tqdm transformers
```
Or with conda:
```bash
conda env create -n {ENV_NAME} -f environment.yaml
```

Now run:

```bash
python qupath_predict.py
```

This script does the following:

* Loads the two models in the current folder (`model_1.pt`, `model_2.pt`).
* Connects to the open slide in QuPath.
* Performs inference to detect:

    * 🟩 **Neoplastic areas** (light green)
    * 🟨 **PTC-only areas** (yellow)
* Overlays these regions directly in the QuPath viewer.

---

## ⚙️ Script Configuration – `qupath_predict.py`

The script comes with many configurable options to suit your needs. Here’s a detailed explanation of each:

---

### 📄 Basic Options

| Argument     | Description                                                                                    | Default      |
|--------------|------------------------------------------------------------------------------------------------|--------------|
| `--file`     | (Optional) Path to a WSI file. If not set, the script uses the slide currently open in QuPath. | `None`       |
| `--device`   | Choose computation device: `'cpu'`, `'cuda'`, `'mps'`, or e.g. `'cuda:0'`.                     | `'cpu'`      |

---

### 🧠 Model Options

| Argument         | Description                                                    | Default        |
|------------------|----------------------------------------------------------------|----------------|
| `--model_path_1` | Path to the first model file (neoplastic detection model).     | `./model_1.pt` |
| `--model_path_2` | Path to the second model file (PTC-only classification model). | `./model_2.pt` |

---

### 🖼️ Image and Tile Settings

| Argument           | Description                                             | Default |
|--------------------|---------------------------------------------------------|---------|
| `--thumbnail_size` | Size of the downsampled image for background detection. | `2048`  |
| `--tile_size_1`    | Size of tiles (in pixels) for the **first model**.      | `96`    |
| `--tile_size_2`    | Size of tiles for the **second model**.                 | `224`   |

---

### 🔲 Region Sampling

| Argument          | Description                                                                                  | Default |
|-------------------|----------------------------------------------------------------------------------------------|---------|
| `--square_side_1` | Side length (in tiles) of the square processed by the **first model**.                       | `4`     |
| `--square_side_2` | Side length (in tiles) for the **second model**.                                             | `2`     |
| `--mpp_1`         | Microns per pixel (resolution) for the **first model**.                                      | `0.97`  |
| `--mpp_2`         | Microns per pixel for the **second model**.                                                  | `0.8`   |
| `--block_scale`   | Scale multiplier to increase the size of the block. If `=1`, the squares are adjacent.       | `1.2`   |
| `--max_splits`    | Maximum number of slide regions to split for processing. Prevents overload with huge slides. | `200`   |

---

### ⚡ Performance and Parallelization

| Argument          | Description                                                                             | Default |
|-------------------|-----------------------------------------------------------------------------------------|---------|
| `--max_workers`   | Number of worker threads used for parallel processing. Higher = faster (if enough CPU). | `10`    |
| `--batch_size_1`  | Batch size for inference using the **first model**.                                     | `512`   |
| `--batch_size_2`  | Batch size for the **second mode**l.                                                    | `128`   |
| `--buffer_size_1` | Internal buffer size (queue) for the **first model**. Higher = faster (if enough RAM)   | `1024`  |
| `--buffer_size_2` | Internal buffer size for the **second model**.                                          | `256`   |

---

### 🧪 Prediction Thresholds

| Argument        | Description                                                               | Default |
|-----------------|---------------------------------------------------------------------------|---------|
| `--threshold_1` | Confidence threshold for the first model to consider a region neoplastic. | `0.8`   |
| `--threshold_2` | Confidence threshold for the second model to consider a region PTC-only.  | `0.5`   |

---

## 🧭 Summary

1. ✅ Start the server in QuPath.
2. ✅ Run `qupath_predict.py`.
3. 🔬 Watch the annotated slide light up with green and yellow predictions.

---

## 💡 Tips

* Want GPU acceleration? Just set `--device cuda` if your machine supports it.
* Don’t forget to adjust `mpp` and `tile_size` values if your custom model expects different resolutions.
* Heatmaps and predictions will be saved in the `--out_path` directory.

---

## 📬 Questions?

If you need help or have questions, feel free to open an issue or contact the project maintainer.

Happy analyzing! 🔬🧠
