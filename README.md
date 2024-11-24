# Bayesian Optimization for Hyperparameter Tuning

This repository implements a Bayesian Optimization workflow for hyperparameter tuning using the Ray Tune framework and ConfigSpace for configuration management. The implementation supports advanced scheduling and search algorithms, including the **BOHB (Bayesian Optimization with HyperBand)** scheduler.

![Learning Rate Selection Over Iterations](/HO_LR_1.PNG)
![Convergence of Validation Accuracy Over Iterations](/Convergence_val_acc.PNG)
![Hyperparameter Trajectories: Learning rate vs. Validation Accuracy](/H_trajectories.PNG)

---

## Table of Contents

- [Features](#features)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Contact](#contact)
- [License](#license)

---

## Features

1. **Bayesian Optimization with BOHB**:
   - Efficient hyperparameter search.
   - Dynamic resource allocation for trials.

2. **Configurable Search Space**:
   - Supports a wide range of hyperparameter configurations.
   - Easily extendable using Hydra and ConfigSpace.

3. **Metrics Monitoring**:
   - Integration with **TensorBoard** for real-time performance tracking.

4. **Supports Imbalanced Datasets**:
   - Metrics tailored for imbalanced datasets to ensure robust evaluation.

5. **Parallelization**:
   - Distributed execution across multiple GPUs and CPUs.

---

## Dependencies

Key dependencies for this project include:

- Python >3.9
- PyTorch (for Vision Transformer models)
- torchvision
- scikit-learn
- ray
- Jupyter Notebook (for data preprocessing)
- Additional dependencies listed in `environment.yaml`.

---
## Contact

If you have any questions, suggestions, or issues, feel free to reach out:

- **Name**: Suraj Bhardwaj
- **Email**: suraj.unisiegen@gmail.com
- **LinkedIn**: [Suraj Bhardwaj](https://www.linkedin.com/in/bhardwaj-suraj)

For project-related inquiries, please use the email mentioned above.

---

## License

This repository is licensed under the [MIT License](https://opensource.org/licenses/MIT) - feel free to modify and distribute the code with proper attribution.
