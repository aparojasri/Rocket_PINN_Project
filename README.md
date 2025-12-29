# Rocket_PINN_Project: High-Fidelity Regenerative Cooling Solver

## 🚀 Overview
This project implements a physics-based computational framework to model the regenerative cooling of liquid rocket engines using supercritical propellants (e.g., RP-1/Dodecane). It serves as the "Ground Truth" generator for data-driven emulators.

The solver resolves the **1D Unsteady Euler Equations** coupled with real-gas thermodynamics to capture the "Pseudo-Boiling" phenomenon—a thermodynamic singularity where specific heat capacity ($C_p$) spikes violently.

## 🧪 Physics Engine
* **Governing Equations:** 1D Conservative Euler System (Mass, Momentum, Energy).
* **Equation of State (EOS):** Helmholtz Energy EOS via **NIST CoolProp**.
* **Turbulence Closure:** Modified Dittus-Boelter correlation embedded in source terms.
* **Numerical Scheme:** 5th-Order WENO reconstruction with SSP-RK3 time integration.

## 📊 Key Features
* **Supercritical Fluid Dynamics:** Accurately resolves density gradients near the critical point ($T_c = 658$ K, $P_c = 1.8$ MPa).
* **Shock Capture:** Handles rapid thermal transients and acoustic waves without oscillation.
* **Data Generation:** Scripts to generate the "Ignition Corpus" (1,200 transient scenarios) for AI training.

## 🛠️ Installation & Requirements
```bash
pip install numpy matplotlib scipy CoolProp
