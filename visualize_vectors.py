"""Plot the embeddings stored in the SQLite search index."""
import matplotlib.pyplot as plt
import numpy as np

from document_database import load_index


def main():
    _, chunks, vectors = load_index()
    if len(chunks) < 2 or vectors.shape[1] < 2:
        print("Index at least two text chunks in FileSense before plotting vectors.")
        return
    print("shape:", vectors.shape)
    print("dtype:", vectors.dtype)

    centered = vectors - vectors.mean(axis=0, keepdims=True)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    points = centered @ components[:2].T

    plt.figure(figsize=(10, 8))
    plt.scatter(points[:, 0], points[:, 1], s=8, alpha=0.5)
    plt.title(f"SQLite embeddings projected to 2D ({len(chunks)} chunks)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()


if __name__ == "__main__":
    main()
