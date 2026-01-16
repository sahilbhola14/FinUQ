from ssgetpy import search
import scipy.io as sio
import numpy as np
import tarfile
import os
import matplotlib.pyplot as plt


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


def is_float16_safe(A):
    """Return True if A can be cast to float16 without inf/NaN."""
    A_cast = np.asarray(A, dtype=np.float16)
    return np.isfinite(A_cast).all()


class IEEEModel:
    def __init__(self, model):
        self.model = model
        assert self.model in ["double", "single", "half"], "invalid model"
        self.precision = self._get_precision()
        self.base = 2
        self.exponent_range = self._get_exponent_range()
        self.urd = 0.5 * self.base ** (1.0 - self.precision)
        self.largest_val = self._largest_representable()

    def _get_precision(self):
        precision = {"double": 53, "single": 24, "half": 11}
        return precision.get(self.model)

    def _get_exponent_range(self):
        exponent_range = {
            "double": [-1021, 1024],
            "single": [-125, 128],
            "half": [-13, 16],
        }
        return exponent_range.get(self.model)

    def _largest_representable(self):
        e_max = self._get_exponent_range()[1]
        precision = self._get_precision()
        x_max = (2.0 - 2.0 ** (-(precision - 1))) * 2.0 ** (e_max - 1)
        return x_max


def get_largest_representable_value(dtype=np.float16):
    types = {"float64": "double", "float32": "single", "float16": "half"}
    model = IEEEModel(types[dtype.__name__])
    return model.largest_val


def is_overflow_save(A, safety_factor=0.95):
    largest_rep = get_largest_representable_value()
    A_cast = np.abs(np.asarray(A, dtype=np.float16))
    row_sum = A_cast.sum(axis=1)
    condition = np.all(row_sum < (largest_rep * safety_factor))
    return condition


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


def load_matrices(file_path):
    """
    Load matrices from a binary file saved by the save_matrices function.

    Args:
        file_path (str): Path to the binary file containing matrices

    Returns:
        list: List of tuples (A, matrix_id, matrix_name, nnz) where:
            - A: numpy array of shape (rows, cols) with dtype=np.float64
            - matrix_id: ID of the matrix
            - matrix_name: Name of the matrix
            - nnz: Number of non-zero elements
    """
    matrices_list = []

    with open(file_path, "rb") as f:
        # Read the number of matrices
        num_matrices = np.fromfile(f, dtype=np.int32, count=1)[0]

        for _ in range(num_matrices):
            # Read dimensions and nnz
            rows = np.fromfile(f, dtype=np.int32, count=1)[0]
            cols = np.fromfile(f, dtype=np.int32, count=1)[0]
            nnz = np.fromfile(f, dtype=np.int32, count=1)[0]

            # Read the matrix data
            matrix_data = np.fromfile(f, dtype=np.float64, count=rows * cols)
            A = matrix_data.reshape((rows, cols), order="C")

            # Note: The original save function doesn't save matrix_id and matrix_name,
            # so we return None for these fields. If you need to preserve them,
            # modify the save function to write them before the matrix data.
            matrix_id = None
            matrix_name = None

            matrices_list.append((A, matrix_id, matrix_name, nnz))

    print(f"Loaded {len(matrices_list)} matrices from {file_path}")
    return matrices_list


def check_matrix_properties(matrices):
    """
    Compute min/max values for each matrix when cast to float64/float32/float16.

    Returns a list of dicts with dtype->(min, max) entries plus overflow info.
    """
    results = []
    dtypes = [np.float64, np.float32, np.float16]
    # dtypes = [np.float64, np.float32]
    for entry in matrices:
        A = entry[0]
        stats = {}
        overflow_dtypes = []
        for dt in dtypes:
            A_cast = np.asarray(A, dtype=dt)
            if not np.isfinite(A_cast).all():
                overflow_dtypes.append(dt.__name__)
            stats[dt.__name__] = (
                np.float64(np.min(A_cast)),
                np.float64(np.max(A_cast)),
            )
        stats["overflow"] = bool(overflow_dtypes)
        stats["overflow_dtypes"] = overflow_dtypes
        results.append(stats)
    return results


def plot_matrix_properties(properties, labels=None, save_path=None, show=False):
    """
    Plot min/max values for each dtype across matrices.

    Args:
        properties (list[dict]): Output from check_matrix_properties.
        labels (list[str] | None): Optional matrix labels for x-axis.
        save_path (str | None): Optional path to save the figure.
        show (bool): Whether to display the plot interactively.
    """
    if not properties:
        raise ValueError("properties is empty")

    dtype_order = ["float64", "float32", "float16"]
    dtype_names = [d for d in dtype_order if d in properties[0]]
    if not dtype_names:
        raise ValueError("No recognized dtypes found in properties")

    x = np.arange(len(properties))
    if labels is None:
        labels = [str(i) for i in x]

    mins = {d: [props[d][0] for props in properties] for d in dtype_names}
    maxs = {d: [props[d][1] for props in properties] for d in dtype_names}

    colors = {
        "float64": "#1B6F6A",
        "float32": "#C74A2A",
        "float16": "#335C81",
    }

    fig, (ax_min, ax_max) = plt.subplots(1, 2, figsize=(10, 4), layout="compressed")
    marker = ["o", "s", "X"]
    for ii, d in enumerate(dtype_names):
        ax_min.scatter(
            x, mins[d], marker=marker[ii], label=d, color=colors.get(d), s=20
        )
        ax_max.scatter(
            x, maxs[d], marker=marker[ii], label=d, color=colors.get(d), s=20
        )

    ax_min.set_title("Per-matrix minimum")
    ax_max.set_title("Per-matrix maximum")
    for ax in (ax_min, ax_max):
        # ax.set_xlabel("Matrix index")
        # ax.set_xticks(x)
        # ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(True, alpha=0.3)
        ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


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

            # Skip matrices that overflow in float16
            if not is_float16_safe(A):
                print("  Skipping: Values overflow in float16")
                continue

            # skip matrix if overflow can occur when performing A * x, where |x| <=1
            if not is_overflow_save(A):
                print(" Skipping: Values in mat-vec can have overflow in float16")
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
    download_square_matrices(n_max=100, size_lim=5000)
    # load the matrix
    file_path = "square_matrix_data/square_matrices.bin"
    if os.path.exists(file_path):
        matrices = load_matrices(file_path)
        for i, (A, matrix_id, matrix_name, nnz) in enumerate(matrices):
            print(f"Matrix {i}: shape={A.shape}, dtype={A.dtype}, nnz={nnz}")
        # check properties
        properties = check_matrix_properties(matrices)
        labels = []
        for i, (_, _, matrix_name, _) in enumerate(matrices):
            labels.append(matrix_name if matrix_name else str(i))
        plot_matrix_properties(
            properties,
            labels=labels,
            save_path="square_matrix_data/matrix_min_max.png",
        )
    else:
        print(f"File not found: {file_path}")
