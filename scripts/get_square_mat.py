from ssgetpy import search
import scipy.io as sio
import numpy as np
import tarfile
import os


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


def save_matrices_bin(
    matrices_list,
    output_file="square_matrices.bin",
    dtype=np.float64,
):
    """
    Save variable-size matrices in a compact binary format.

    Layout:
      int32 num_matrices
      for each matrix:
        int32 rows
        int32 cols
        int32 nnz
        rows*cols values (dtype, row-major)
    """

    save_dir = "square_matrix_data"
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, output_file)

    with open(path, "wb") as f:
        np.int32(len(matrices_list)).tofile(f)

        for A, matrix_id, matrix_name, nnz in matrices_list:
            A = np.asarray(A, dtype=dtype, order="C")

            # safety checks
            if not np.isfinite(A).all():
                raise ValueError("NaN or Inf found in matrix")
            rows, cols = A.shape
            assert nnz <= rows * cols, "invalid number of nnz."
            np.int32(rows).tofile(f)
            np.int32(cols).tofile(f)
            np.int32(nnz).tofile(f)
            # raw contiguous write
            A.ravel(order="C").tofile(f)

    print(f"Saved {len(matrices_list)} matrices to {path}")


def download_square_matrices(n_max=10, output_format="json", size_lim=5000):
    """Download n_max square matrices with real values."""
    print(f"Searching for {n_max} square matrices...\n")

    # Search for matrices
    results = search(limit=10000)

    # Filter for square matrices
    square_matrices = []
    for ii, matrix in enumerate(results):
        try:
            rows = int(matrix.rows)
            cols = int(matrix.cols)
        except (TypeError, ValueError):
            continue
        if (rows == cols) and rows <= size_lim:
            square_matrices.append(matrix)
        if len(square_matrices) >= n_max:
            break
    print(
        f"Found {len(square_matrices)}/{len(results)} square matrices "
        f"of size less than {size_lim}\n"
    )

    # Download and process each matrix
    real_matrices = []
    for i, matrix in enumerate(square_matrices, 1):
        print(f"Processing Matrix {i}/{len(square_matrices)}: {matrix.name}")
        print(f"  ID: {matrix.id}")
        print(f"  Non-zeros: {matrix.nnz}")

        try:
            # Download matrix
            filepath = download_matrix(matrix)
            print(f"  Downloaded: {filepath}")

            # Load matrix
            A = load_matrix(filepath)
            assert A.shape[0] == A.shape[1]
            assert A.shape[0] <= size_lim
            print(f"  Matrix shape: {A.shape}")
            print(f"  Data type: {A.dtype}")

            # Skip complex matrices
            if np.iscomplexobj(A):
                print("  Skipping: Matrix is complex-valued")
                continue

            # Store real matrix
            real_matrices.append((A, matrix.id, matrix.name, int(np.count_nonzero(A))))
            print(f"{matrix.name} Stored successfully")

        except Exception as e:
            print(f"  Error: {e}")

    # Save all matrices in specified format
    print(f"Found {len(real_matrices)} real matrices")
    save_matrices_bin(real_matrices)


if __name__ == "__main__":
    download_square_matrices(n_max=10, size_lim=5000)
