# RBF-MAT: Computing Medial Axis Transform from Point Clouds by Optimizing Radial Basis Functions

## 📄 Project Overview

**RBF-MAT** is a method for computing the **Medial Axis Transform (MAT)** from point clouds using **Radial Basis Functions (RBFs)**. This approach involves selecting initial medial spheres from Voronoi vertices, optimizing their centers and radii iteratively, and constructing connectivity using a restricted power diagram.

### Key Features

- **Medial Axis Transform (MAT) Computation** from point clouds.
- **Radial Basis Functions (RBFs)** for surface reconstruction.
- **Restricted Power Diagram** for refining connectivity.
- Accurate approximation of point cloud surfaces.

---

## 📂 Project Structure

- **`matcal`**: Main file for MAT computation.
- **`medial_axis_approx.py`**: Preprocessing input point clouds.
- **`solver.py`**: Optimizing centers and radii of medial spheres.
- **`utils.py`**: other function like normalize point.

---

## 🛠️ Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/your-username/RBF-MAT.git
    cd RBF-MAT
    ```

2. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

    Ensure you have the following libraries:
    - `numpy`
    - `scipy`
    - `matplotlib`
    - `open3d`
    - `PyTorch`

---

## 🚀 Usage




python matcal.py 
