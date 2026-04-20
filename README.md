# Physics-Informed Bayesian Modeling for LPBF MS300 Maraging Steel

> Predicting **relative density** and **residual stress** in Laser Powder Bed Fusion (LPBF) using Eagar–Tsai thermal physics combined with Gaussian Process Regression.

---

## Project Overview

This project develops and benchmarks physics-informed machine learning models for predicting two critical quality metrics in LPBF-manufactured MS300 maraging steel parts:

- **Relative Density** — porosity-related quality indicator
- **Residual Stress** — internal stress state after solidification

The core idea is a **hierarchical Bayesian framework** where physics-based priors (Eagar–Tsai melt pool model + VED sigmoid fits) are combined with Gaussian Process Regression (GPR) to correct residuals that pure physics cannot capture.

---

## Development Timeline

The project was built in four phases, first establishing baselines and physics models for relative density, then repeating the same progression for residual stress:

```
Phase 1 ── Relative density: baseline models
           BasicModels_RelDensity.ipynb
           Linear & polynomial regression (deg 1–5), Decision Tree, Random Forest
           │
Phase 2 ── Relative density: Eagar–Tsai physics-informed model
           EagerTsai_based_bayesian_RelDensity.ipynb
           ET melt pool + VED sigmoid prior + 3-layer GPR pipeline
           │
           ▼  [Pivot: target property → Residual Stress]
           │
Phase 3 ── Residual stress: baseline models
           BasicModels_ResStress.ipynb
           Linear & polynomial regression (deg 1–5), Decision Tree, Random Forest
           │
Phase 4 ── Residual stress: Eagar–Tsai physics-informed model
           EagerTsai_based_bayesian_MS300_ResStress.ipynb
           Full physics-informed M5 model with uncertainty quantification
```

This progression allowed the methodology developed for relative density to be validated and then transferred to the harder residual stress prediction problem.

---

## Repository Structure

```
Explo/
├── EagerTsai_based_bayesian_MS300_ResStress.ipynb   # Main notebook: residual stress modeling
├── EagerTsai_based_bayesian_RelDensity.ipynb        # Main notebook: relative density modeling
├── BasicModels_ResStress.ipynb                      # Baseline models for residual stress
├── BasicModels_RelDensity.ipynb                     # Baseline models for relative density
└── residual_stress_maraging_steel_dataset.xlsx      # Experimental dataset (MS300 LPBF)
└── MS300Fixed.xlsx                                  # Experimental dataset for Density(MS300 LPBF)

```

---

## Methodology

### Input Parameters (Process Variables)
| Parameter | Symbol | Unit |
|-----------|--------|------|
| Laser Power | P | W |
| Scan Speed | v | mm/s |
| Hatch Spacing | h | mm |

### Physics Layer 1 — Eagar–Tsai Thermal Model
Computes melt pool **width** and **depth** from first principles using:
- Material properties of MS300 (density, thermal conductivity, specific heat, absorptivity)
- Numerical integration of the moving Gaussian heat source solution
- Bisection method to find melt pool boundary at liquidus temperature (~1410 °C)

### Physics Layer 2 — VED Prior
Fits a sigmoid curve to the relationship between Volumetric Energy Density (VED) and the target property:

```
VED = P / (v × h × t)   [J/mm³]
```

Three sigmoid variants (simple, bell-shaped, double) are evaluated; the best R² model is selected as the physics prior.

### ML Layer — Gaussian Process Regression (GPR)
A three-layer GPR pipeline:
1. **Layer 1**: Corrects ET-predicted melt pool width
2. **Layer 2**: Corrects ET-predicted melt pool depth (conditioned on corrected width)
3. **Layer 3**: Predicts the target property residual (actual − VED prior), then adds the prior back

The GPR kernel is an additive combination of `RBF + DotProduct + WhiteKernel`, capturing both nonlinear physics and linear trends, plus noise.

### Models Compared
| Model | Description |
|-------|-------------|
| **Linear** | Ordinary linear regression (degree 1 polynomial) |
| **Polynomial (deg 2–5)** | Polynomial regression up to degree 5 |
| **Decision Tree** | Decision Tree Regressor |
| **Random Forest** | Random Forest Regressor (ensemble of decision trees) |
| **M3** | GPR with ET melt pool width as proxy feature (no physics prior) |
| **M5** | Full physics-informed GPR (ET geometry + VED prior + 3-layer pipeline) |

---

## Evaluation

Models are benchmarked using **250× repeated 70/30 train-test cross-validation** with the following metrics:

| Metric | Description | Goal |
|--------|-------------|------|
| MAE | Mean Absolute Error | ↓ Lower is better |
| RMSE | Root Mean Square Error | ↓ Lower is better |
| Spearman Rₛ | Rank correlation | ↑ Higher is better |
| Kendall τ | Rank concordance | ↑ Higher is better |

Results are reported as **Median (IQR)** across all CV runs for robustness.

---

## Getting Started

### Requirements

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn openpyxl
```

### Running the Notebooks

1. Place `residual_stress_maraging_steel_dataset.xlsx` in the same directory as the notebooks.
2. Open and run cells in order:
   - Start with `EagerTsai_based_bayesian_MS300_ResStress.ipynb` for residual stress
   - Or `EagerTsai_based_bayesian_RelDensity.ipynb` for relative density
   - `BasicModels_*.ipynb` contain simpler baseline models for comparison

### Making Predictions on New Parameters

After running all training cells, use the interactive prediction cell at the end of each main notebook:

```
Enter Laser Power   (W)   [e.g. 300]:  ___
Enter Scan Speed    (mm/s) [e.g. 1000]: ___
Enter Hatch Spacing (mm)  [e.g. 0.09]: ___
```

The cell will output:
- ET melt pool width and depth (raw + GPR-corrected)
- W/D ratio with keyholing risk flag
- Predicted residual stress / density with uncertainty (±1σ)
- 95% confidence interval
- Visualization: melt pool cross-section + prediction in dataset context

---

## Output Files Generated

| File | Description |
|------|-------------|
| `EDA.png` | Exploratory data analysis plots |
| `ET_Model_MS300.png` | Eagar–Tsai melt pool predictions across dataset |
| `VED_Prior_MS300.png` | Sigmoid prior fits and residual distribution |
| `PredVsActual_Stress.png` | Predicted vs actual scatter for M3 and M5 |
| `CV_Results_MS300.csv` | Full cross-validation benchmarking results |
| `CustomPrediction_Stress.png` | Visualization for user-defined input parameters |

---

## Key Findings

- **M5 consistently outperforms M3** in both error metrics (MAE, RMSE) and ranking metrics (Spearman Rₛ, Kendall τ)
- Physics priors are most beneficial in **low-data regions** of the process space
- The Eagar–Tsai model captures the **process → thermal → microstructure** relationship effectively
- GPR uncertainty estimates are well-calibrated, providing reliable confidence intervals

---

## Material Properties Used (MS300 Maraging Steel)

| Property | Value |
|----------|-------|
| Liquidus Temperature | ~1410 °C |
| Thermal Conductivity | 25 W/m·K |
| Density | 8000 kg/m³ |
| Specific Heat | 460 J/kg·K |
| Laser Absorptivity | 0.40 |
| Beam Radius | 35 µm |

---

## Citation

If you use this work, please cite the dataset and acknowledge the Eagar–Tsai thermal model:

> Eagar, T.W. & Tsai, N.S. (1983). *Temperature fields produced by traveling distributed heat sources.* Welding Journal, 62(12), 346–355.

**EagerTsai Bayesian MS300 Project**  
Physics-informed machine learning for additive manufacturing process optimization.
