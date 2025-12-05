
🎵 Music Genre Classification using Audio Features & Random Forest

This project builds a simple and effective Music Genre Classifier using Python, Librosa, and Scikit-learn. It supports four genres — Classical, Blues, HipHop, and Rock — stored inside a structured music_dataset/ directory. Each .wav audio file is processed using Librosa to extract key features such as MFCCs, Chroma, Spectral Centroid, and Spectral Rolloff. These features are averaged and combined to form the input dataset.

The script then splits the data into training and testing sets and trains a RandomForestClassifier with 200 decision trees. After training, the model predicts genres for the test samples and calculates the overall classification accuracy.

To visualize performance, the program automatically generates and saves PNG outputs including:

accuracy.png – overall model accuracy

confusion_matrix.png – genre-wise performance

feature_distribution.png – average feature values


This project is great for learning audio processing and ML classification.


---

If you want a 100-word or more formal version, just tell me!
