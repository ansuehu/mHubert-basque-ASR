import numpy as np
import joblib
from sklearn.cluster import MiniBatchKMeans

def train_kmeans(feature_file, n_clusters=1000, batch_size=10000):
    all_features = np.load(feature_file)
    print(f"{all_features.shape}")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=0, verbose=1)
    kmeans.fit(all_features)
    kmeans_file = "/home/andoni.sudupe/mHubert_finetune/checkpoints/kmeans/basque_hubert_k1000_L9.pt"
    joblib.dump(kmeans, kmeans_file)
    print(f"KMeans model trained and saved as {kmeans_file}")

def main():
    feature_file = "/home/andoni.sudupe/mHubert_finetune/data/composite_eu/hubert_basque_features.npy"
    train_kmeans(feature_file)

if __name__ == "__main__":
    main()