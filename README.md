# Handwritten Character and Digit Recognition (0–9 + A–Z)

A beginner-friendly Deep Learning project using a CNN trained on:
- MNIST: handwritten digits 0–9
- EMNIST Letters: handwritten uppercase letters A–Z

The model predicts one of 36 classes: `0–9` and `A–Z`.

## Important: first run the notebook
The ZIP contains the complete source code, but **not a trained model**. Training creates `models/handwritten_cnn.keras`.
This avoids shipping a large model file and lets you train your own model in free Google Colab.

## 1. Project structure

```text
handwritten_recognition_project/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── notebooks/
│   └── Handwritten_Character_Digit_Recognition.ipynb
├── models/
│   └── (trained model is saved here)
├── sample_images/
│   └── (put test images here)
└── src/
    └── labels.py
```

## 2. Train in Google Colab (recommended)

1. Extract this ZIP on your computer.
2. Open `notebooks/Handwritten_Character_Digit_Recognition.ipynb`.
3. Upload the notebook to Google Colab: https://colab.research.google.com/
4. In Colab, choose **Runtime → Change runtime type → T4 GPU** if available. CPU also works, but may take longer.
5. Run the notebook cells from top to bottom.
6. At the end, download `handwritten_cnn.keras` from Colab's Files panel, then place it in this project's `models/` folder.

The notebook downloads MNIST and EMNIST Letters automatically using TensorFlow Datasets. The first run needs an internet connection and may take a few minutes to download the datasets.

## 3. Run the web app in VS Code

Open the extracted project folder in VS Code. In the VS Code terminal:

### Windows
```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown in the terminal (usually http://localhost:8501).

If you have not copied the trained model into `models/handwritten_cnn.keras`, the app will explain that the model is missing.

## 4. How to use the app

- Upload a clear image containing one handwritten digit or one uppercase letter.
- Use dark ink on a light background or light ink on a dark background; the app tries to normalize polarity automatically.
- Crop the image so the character is reasonably centered and fills most of the image.
- The app displays the predicted class and confidence.

## 5. Model details

- Input: 28 × 28 grayscale image
- Classes: 36 (digits 0–9, uppercase A–Z)
- Architecture: Convolutional Neural Network (CNN)
- Dataset: MNIST + EMNIST Letters
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy

## 6. Troubleshooting

- **TensorFlow installation issue:** use Python 3.10 or 3.11 in a fresh virtual environment. TensorFlow support can vary by Python version and operating system.
- **Colab runtime disconnects:** save/download the model after training. You can reconnect and rerun cells if needed.
- **Model file not found:** ensure the file is named exactly `handwritten_cnn.keras` and is inside `models/`.
- **Poor predictions:** use a single centered character, avoid shadows, and make the character bold and clear.
- **EMNIST letter appearance:** some letters can be visually similar (for example I/J or O/Q). This is a dataset limitation, not necessarily a coding error.

## 7. Academic note

Report the test accuracy separately for digits and letters, and do not claim a particular accuracy before running the notebook. Results depend on training settings and the data split.
