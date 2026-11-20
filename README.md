
# Human Detection

A Python and OpenCV project for detecting people in camera footage. The repository preserves the original notebook alongside the updated standalone application.

## Versions

| Version | Entry point | Input | Features |
| --- | --- | --- | --- |
| **Human Detection V1** | [`Human Detection V1.ipynb`](Human%20Detection%20V1.ipynb) | Webcam | YOLOv3, HOG, Haar cascade, person tracking, and on-screen statistics |
| **Human Detection V2** | [`Human Detection V2.py`](Human%20Detection%20V2.py) | Webcam or local video file | Person detection and tracking, configurable input, and optional age and facial expression estimates |

**Human Detection V1** is the original implementation and remains unchanged. Use **Human Detection V2** for new installations or video-file testing.

## System Requirements and Model Setup (Windows / VS Code)

1. Install 64-bit Python 3.12 and open this project folder in VS Code.
2. If you want YOLOv3 person detection, download [`yolov3.weights`](https://pjreddie.com/media/files/yolov3.weights) and place it in the repository root, next to the included `yolov3.cfg` and `coco.names`. Both versions can fall back to HOG without the weights. **Version 1 attempts to download missing YOLO files automatically**, so download the weights beforehand if you want to avoid that startup attempt.
3. Use a local desktop session with camera access. Both versions use an OpenCV window, not a browser preview. If Windows blocks camera access, check **Settings → Privacy & security → Camera** and close other apps using the camera.

Use separate virtual environments for the two versions. The commands below call each environment's Python directly, so PowerShell activation is not required.

## Human Detection V1 — Original Notebook

1. Install the **Python** and **Jupyter** extensions in VS Code.
2. From the repository root, create the environment and install the notebook dependencies:

   ```powershell
   py -3.12 -m venv .venv-v1
   .\.venv-v1\Scripts\python.exe -m pip install --upgrade pip
   .\.venv-v1\Scripts\python.exe -m pip install -r requirements.txt requests ipykernel
   ```

3. Open `Human Detection V1.ipynb`. Click **Select Kernel → Python Environments** and select `.venv-v1\Scripts\python.exe`. Run the notebook from the repository root so its relative model paths resolve correctly.
4. Select **Run All**. The notebook uses camera index `0`. Focus the OpenCV window and press **Q** to stop it before running Version 2.

## Human Detection V2 — Standalone Application

1. Create a fresh environment:

   ```powershell
   py -3.12 -m venv .venv-v2
   .\.venv-v2\Scripts\python.exe -m pip install --upgrade pip
   ```

2. Choose **one** installation option:

   ```powershell
   # Full setup: NumPy, desktop OpenCV, DeepFace, tf-keras and their dependencies
   .\.venv-v2\Scripts\python.exe -m pip install -r requirements-face.txt

   # OR: Person detection only (NumPy and desktop OpenCV)
   .\.venv-v2\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. Run the script with the option that matches your installation:

   ```powershell
   # Full setup: age and expression estimates are enabled by default
   .\.venv-v2\Scripts\python.exe ".\Human Detection V2.py" --camera 0

   # Person-only setup: turn off face analysis
   .\.venv-v2\Scripts\python.exe ".\Human Detection V2.py" --no-face-attributes

   # Local video instead of the webcam (add --no-face-attributes if needed)
   .\.venv-v2\Scripts\python.exe ".\Human Detection V2.py" --video "C:\Videos\test.mp4"
   ```

   Keep the quotes around the filename because it contains spaces. Try `--camera 1` if camera `0` is not the camera you want. Press **Q** in the preview window to quit.

### Age and expression model files

On first run, DeepFace loads both models before opening the camera and downloads missing weights. If the automatic download fails, create the default Windows weights folder:

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.deepface\weights"
```

Download both files into that folder with their exact names, then restart the script:

- [`age_model_weights.h5`](https://github.com/serengil/deepface_models/releases/download/v1.0/age_model_weights.h5)
- [`facial_expression_model_weights.h5`](https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5)

If you set `DEEPFACE_HOME`, use `.deepface/weights` under that directory instead. Use good lighting and face the camera directly. Age is approximate, and expression labels describe facial appearance, not how someone feels. Use the feature with consenting people; the script does not save or upload face images.

## Preview Controls

Focus the OpenCV preview window before pressing a key. These controls are available in both versions.

| Key | Action |
| --- | --- |
| `Q` | Quit the preview |
| `Y` | Switch to YOLOv3 if the model loaded successfully |
| `H` / `A` | Switch to HOG / Haar cascade |
| `T` / `C` / `S` | Toggle trajectories / confidence display / statistics |
| `R` | Reset tracking |
| `+` / `-` | Raise / lower the YOLOv3 confidence threshold |

The cumulative detection count adds detections across frames and does not represent unique visitors. Face analysis in V2 runs independently of person tracking, so face overlays can lag during fast movement.

## Troubleshooting

- **`No module named cv2`**: Install into the environment used to run the script, or select that environment as the notebook kernel. The package name is `opencv-python`; the import name is `cv2`.
- **`imshow` is not implemented or `readNetFromDarknet` is missing**: Use the pinned desktop OpenCV wheel. OpenCV distributions share the `cv2` namespace, so remove competing wheels first. For Version 1, replace `.venv-v2` with `.venv-v1` below and restart the notebook kernel afterward.

  ```powershell
  .\.venv-v2\Scripts\python.exe -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless
  .\.venv-v2\Scripts\python.exe -m pip install "opencv-python==4.12.0.88"
  .\.venv-v2\Scripts\python.exe -c "import cv2; print(cv2.__version__, hasattr(cv2.dnn, 'readNetFromDarknet'), cv2.currentUIFramework())"
  ```

  Expected on Windows: `4.12.0 True WIN32`.
- **YOLO falls back to HOG**: Check that `yolov3.weights` finished downloading and is next to `yolov3.cfg`, `coco.names`, and the Version 2 script. For Version 1, also check the notebook's working directory.
- **TensorFlow prints oneDNN or deprecation messages**: These are not fatal if startup continues to `Face analysis ready.`

## What to Commit to GitHub

Commit the two versioned source files, `README.md`, `requirements.txt`,
`requirements-face.txt`, `.gitignore`, `.gitattributes`, `coco.names`, and
`yolov3.cfg`. The last two files are small configuration assets, not weights.

Do **not** commit `yolov3.weights`, `age_model_weights.h5`, or
`facial_expression_model_weights.h5`. Download them separately using the links
above. Keep `yolov3.weights` in the project root and the two DeepFace `.h5` files
in the DeepFace weights folder, not in the project root. The `.gitignore` rules
also exclude Python caches and the local environments. Keep personal test
footage outside the repository.

If the weights are already staged, remove them from the Git index without
deleting the local files, then check what will be committed:

```powershell
git rm --cached --ignore-unmatch -- yolov3.weights age_model_weights.h5 facial_expression_model_weights.h5
git status --short
```

If a large file is already in an earlier commit, `.gitignore` and unstaging do
not remove it from history. Remove it from the affected commits before pushing
if GitHub rejects the push for exceeding its file size limit.