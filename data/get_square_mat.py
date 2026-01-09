from ssgetpy import search
import scipy.io as sio
import numpy as np
import tarfile
import os
import json


def extract_mtx_file(tar_gz_path):
    """Extract .mtx file from tar.gz archive."""
    extract_dir = os.path.dirname(tar_gz_path)

    with tarfile.open(tar_gz_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    # Find the extracted .mtx file
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.endswith(".mtx"):
                return os.path.join(root, file)

    raise FileNotFoundError("No .mtx file found in archive")


def download_matrix(matrix):
    """Download matrix and return file path."""
    mat_filepath = matrix.download(format="MM")

    if isinstance(mat_filepath, tuple):
        mat_filepath = mat_filepath[0]

    if mat_filepath.endswith(".tar.gz"):
        mat_filepath = extract_mtx_file(mat_filepath)

    return mat_filepath


def load_matrix(filepath):
    """Load matrix from file and convert to dense array."""
    A = sio.mmread(filepath)

    if hasattr(A, "toarray"):
        return A.toarray()
    else:
        return np.array(A)


def save_matrices_binary(matrices_list, output_file="matrices.npy"):
    """Save list of matrices as numpy binary file."""
    # Pad matrices to same size or store as object array
    np.save(output_file, np.array(matrices_list, dtype=object), allow_pickle=True)
    return output_file


def save_matrices_json(matrices_list, output_file="matrices.json"):
    """Save matrices as JSON for easy parsing in C++."""

    data = {"num_matrices": len(matrices_list), "matrices": []}

    for i, (A, matrix_id, matrix_name) in enumerate(matrices_list):
        data["matrices"].append(
            {
                "id": int(matrix_id),
                "name": matrix_name,
                "rows": int(A.shape[0]),
                "cols": int(A.shape[1]),
                "data": A.flatten().tolist(),  # Flattened 1D array
            }
        )

    with open(output_file, "w") as f:
        json.dump(data, f)

    return output_file


def download_square_matrices(n_max=10, output_format="json"):
    """Download n_max square matrices with real values."""
    print(f"Searching for {n_max} square matrices...\n")

    # Search for matrices
    results = search()

    # Filter for square matrices
    square_matrices = []
    for matrix in results:
        # Filter for square matrices only
        if matrix.rows == matrix.cols:
            square_matrices.append(matrix)
            if len(square_matrices) >= n_max:
                break

    print(f"Found {len(square_matrices)} square matrices\n")

    # Download and process each matrix
    real_matrices = []
    for i, matrix in enumerate(square_matrices, 1):
        assert matrix.rows == matrix.cols
        print(f"Processing Matrix {i}/{len(square_matrices)}: {matrix.name}")
        print(f"  ID: {matrix.id}")
        print(f"  Non-zeros: {matrix.nnz}")

        try:
            # Download matrix
            filepath = download_matrix(matrix)
            print(f"  Downloaded: {filepath}")

            # Load matrix
            A = load_matrix(filepath)
            print(f"  Matrix shape: {A.shape}")
            print(f"  Data type: {A.dtype}")

            # Skip complex matrices
            if np.iscomplexobj(A):
                print("  Skipping: Matrix is complex-valued")
                continue

            # Store real matrix
            real_matrices.append((A, matrix.id, matrix.name))
            print(f"{matrix.name} Stored successfully")

        except Exception as e:
            print(f"  Error: {e}")

    # Save all matrices in specified format
    if output_format == "json":
        output_file = save_matrices_json(
            real_matrices, output_file="square_matrix_data.json"
        )
        print(f"Saved {len(real_matrices)} matrices to: {output_file}")
    elif output_format == "npy":
        output_file = save_matrices_binary(
            real_matrices, output_file="square_matrix_data.npy"
        )
        print(f"Saved {len(real_matrices)} matrices to: {output_file}")


if __name__ == "__main__":
    download_square_matrices(n_max=1)
