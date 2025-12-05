import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

# ------------------------- CONFIG -------------------------
DATASET_DIR = "music_dataset"
GENRES = ["classical", "blues", "hiphop", "rock"]
SAMPLE_RATE = 22050

# Folder to save PNG outputs
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    # Convert to mean values
    features = np.hstack([
        np.mean(mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.mean(spectral_centroid),
        np.mean(spectral_rolloff)
    ])

    return features


# ------------------ LOAD DATASET ------------------
X, y = [], []

print("\n📥 Loading Dataset...\n")

for genre in GENRES:
    path = os.path.join(DATASET_DIR, genre)
    for file in os.listdir(path):
        if file.endswith(".wav"):
            file_path = os.path.join(path, file)
            print(f"Extracting features → {file_path}")
            features = extract_features(file_path)
            X.append(features)
            y.append(genre)

X = np.array(X)
y = np.array(y)

print("\n✅ Dataset Loaded!")


# ------------------ TRAIN / TEST SPLIT ------------------
print("\n📊 Splitting Dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print("✅ Split Done!")


# ------------------ MODEL TRAINING ------------------
print("\n🚀 Training Model...")
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)
print("🎉 Training Completed!\n")


# ------------------ EVALUATION ------------------
print("🔍 Evaluating Model...")

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n🎯 Accuracy: {acc * 100:.2f}%\n")


# ------------------ SAVE ACCURACY GRAPH ------------------
plt.figure(figsize=(6, 4))
plt.bar(["Accuracy"], [acc])
plt.ylim(0, 1)
plt.title("Model Accuracy")
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy.png"))
plt.close()

print("📁 Saved: outputs/accuracy.png")


# ------------------ CONFUSION MATRIX ------------------
cm = confusion_matrix(y_test, y_pred, labels=GENRES)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
            xticklabels=GENRES, yticklabels=GENRES)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

print("📁 Saved: outputs/confusion_matrix.png")


# ------------------ FEATURE DISTRIBUTION PLOT ------------------
plt.figure(figsize=(10, 5))
plt.plot(np.mean(X, axis=0))
plt.title("Average Feature Values Across Dataset")
plt.xlabel("Feature Index")
plt.ylabel("Value")
plt.savefig(os.path.join(OUTPUT_DIR, "feature_distribution.png"))
plt.close()

print("📁 Saved: outputs/feature_distribution.png")

print("\n🎉 ALL PNG OUTPUTS SAVED SUCCESSFULLY inside /outputs/")
