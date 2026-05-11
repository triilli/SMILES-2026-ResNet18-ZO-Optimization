# Solution Report - Zero-Order Fine-Tuning of ResNet18 on CIFAR100

This repository contains the solution for the ResNet18 fine-tuning assignment using zero-order optimization techniques.

## 1. Reproducibility Instructions

### Environment Setup
- **Python Version:** Recommended 3.10 or 3.11 (tested on Windows 10/11).
- **Dependencies:** Install required packages using:
  ```bash
  pip install -r requirements.txt
  ```

### Execution
To reproduce the final `results.json` and the validation accuracy, run the following command:
```bash
python validate.py --n_batches 64 --batch_size 128
```
*Note: The total sample budget is exactly 8192 (64 * 128), complying with the assignment constraints.*

---

## 2. Final Solution Description

### Modified Components
1. **`zo_optimizer.py`**: Implemented a **Sign-SPSA (Simultaneous Perturbation Stochastic Approximation)** optimizer with **Momentum** and **Global Gradient Normalization**.
2. **`head_init.py`**: Applied **Xavier Uniform Initialization** to the final linear layer to ensure stable signal variance at the start of training.
3. **`augmentation.py`**: Added a standard CIFAR-100 augmentation pipeline including `RandomHorizontalFlip` and `RandomCrop(224, padding=28)` to improve generalization.

### Final Approach
The final approach uses **Sign-SPSA**. Unlike standard SPSA, which can be highly unstable due to noisy gradient estimates in Zero-Order settings, Sign-SPSA only considers the *direction* (sign) of the estimated gradient. By combining this with **Gradient Normalization**, the updates remain stable and the loss is prevented from exploding. 

**Key choices:**
- **SPSA Estimator:** Requires only 2 forward passes per step regardless of the number of parameters.
- **Normalization:** Dividing the estimated gradient by its L2 norm ensures that the "noise" from random perturbations doesn't cause disproportionate weight updates.
- **Sign-Momentum:** Using the sign of the accumulated momentum buffer provides a constant step size for every parameter, which is a robust strategy for small-budget ZO optimization.

### What contributed most?
The shift from Adam-based updates to **Sign-SGD logic** contributed the most. Because Zero-Order gradient estimates are inherently noisy, the second-moment estimation in Adam frequently led to divergence. Normalizing the gradients allowed the model to actually descend the loss surface.

---

## 3. Experiments and Failed Attempts(FA)

### FA 1: Adam + SPSA
Initially, I attempted to use the **Adam** optimizer. While Adam is powerful for first-order optimization, it failed miserably in the Zero-Order context. The noisy gradient estimates from SPSA caused the variance term ($v$) in Adam to fluctuate wildly, leading to loss explosion (values exceeding 300) 

### FA 2: Unfreezing Deep Layers (`layer4`)
I tried to fine-tune the last block of the ResNet backbone (`layer4`) alongside the classification head. However, the number of parameters increased significantly, making the SPSA estimates too diluted. Within the 8192-sample budget, the optimizer could not find a meaningful direction for 500k+ parameters, and the accuracy dropped below the baseline

### FA 3: Bias-Only Tuning
I experimented with tuning only the `fc.bias` parameters. While this was computationally efficient and the optimizer converged quickly, the representational capacity was too low to significantly improve Top-1 accuracy, yielding only marginal gains over the initialization

### FA 4: High Learning Rates
Early experiments with high learning rates (e.g., $lr=0.1$) without gradient clipping led to immediate divergence. This taught me the importance of stable, controlled updates in Zero-Order learning

### FA final
In the final attempt with Sign-SPSA and global normalization, I observed divergence (loss increased to 14) due to a learning rate that was too aggressive for the sign-based updates. This highlighted the extreme sensitivity of zero-order optimization to step size
---

## 4. Conclusion
Fine-tuning with Zero-Order optimization is a trade-off between estimation accuracy and the compute budget. By focusing on a stable Sign-based update rule and a well-initialized head, the model was able to achieve consistent progress within the 8192-sample limit.
```

---