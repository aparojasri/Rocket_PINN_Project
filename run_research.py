import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# --- PART 1: PHYSICS DATA GENERATION (The Bartz Simulation) ---
def generate_rocket_data(num_samples=5000):
    print(f"🚀 GENERATING {num_samples} PHYSICS SAMPLES...")
    data = []
    for _ in range(num_samples):
        # 1. Inputs (Operating Conditions)
        P_c = np.random.uniform(50, 100)       # Chamber Pressure (bar)
        MR = np.random.uniform(2.2, 2.8)       # Mixture Ratio
        mdot = np.random.uniform(0.5, 1.5)     # Coolant Flow (kg/s)
        
        # 2. Physics Model (Simplified Bartz Logic)
        # Gas Temp increases with Pressure
        T_gas = 3300 + (P_c - 50) * 12 + (MR - 2.2) * 200
        # Heat Transfer Coefficient scales with Pressure^0.8
        h_g = 0.025 * (P_c ** 0.8)
        
        # 3. Inject Coking Fault (Anomaly)
        if np.random.random() < 0.2: # 20% chance of fault
            coking_factor = np.random.uniform(0.2, 0.9) # Insulating layer
        else:
            coking_factor = 0.0 # Clean wall
            
        # 4. Calculate Wall Temp (Equilibrium)
        # Cooling improves with flow, degrades with coking
        cooling_efficiency = (mdot * 5000) / (1 + coking_factor * 5)
        T_wall = (h_g * T_gas + cooling_efficiency * 300) / (h_g + cooling_efficiency)
        
        # Add realistic sensor noise
        T_wall += np.random.normal(0, 5) 
        
        data.append([P_c, MR, mdot, T_wall])
        
    cols = ['Pressure', 'MixtureRatio', 'FlowRate', 'WallTemp']
    return pd.DataFrame(data, columns=cols)

# --- PART 2: THE PINN MODEL ARCHITECTURE ---
class RocketPINN(nn.Module):
    def __init__(self):
        super().__init__()
        # 4-Layer Deep Neural Network
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(), # Tanh is crucial for Physics Gradients
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1) # Output: Wall Temperature
        )
        
    def forward(self, x):
        return self.net(x)

# --- PART 3: EXECUTION SCRIPT ---
if __name__ == "__main__":
    # A. Generate Data
    df = generate_rocket_data()
    
    # B. Prepare for AI
    X = df[['Pressure', 'MixtureRatio', 'FlowRate']].values
    y = df[['WallTemp']].values
    
    # Scale Data (Crucial for AI convergence)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # Convert to PyTorch Tensors
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y_scaled)
    
    # Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2)
    
    # C. Train Model
    print("🧠 STARTING TRAINING...")
    model = RocketPINN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    epochs = 2000
    loss_history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.5f}")
            
    # D. Evaluate & Visualize
    print("✅ TRAINING COMPLETE. EVALUATING...")
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_test)
        # Convert back to real units (Kelvin)
        preds_real = scaler_y.inverse_transform(preds_scaled.numpy())
        actual_real = scaler_y.inverse_transform(y_test.numpy())
        
    # Calculate Accuracy (RMSE)
    rmse = np.sqrt(np.mean((preds_real - actual_real)**2))
    print(f"🎯 FINAL ACCURACY (RMSE): {rmse:.2f} Kelvin")
    
    # Save Graph for Paper
    plt.figure(figsize=(8, 6))
    plt.scatter(actual_real, preds_real, alpha=0.5, color='blue', s=10)
    plt.plot([actual_real.min(), actual_real.max()], [actual_real.min(), actual_real.max()], 'r--', lw=2)
    plt.xlabel('Actual Physics Temp (K)')
    plt.ylabel('AI Predicted Temp (K)')
    plt.title(f'PINN Model Performance\nRMSE: {rmse:.2f} K')
    plt.grid(True)
    plt.savefig('pinn_results.png')
    print("📊 Graph saved as 'pinn_results.png'")