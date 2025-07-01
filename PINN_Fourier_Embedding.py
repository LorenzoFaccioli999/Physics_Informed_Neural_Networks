# =============================================================================
# PINN with Fourier Embedding - Modular, User-Friendly Version
# =============================================================================
# This script implements a Physics-Informed Neural Network (PINN) with Fourier embedding.
# It is structured for clarity, modularity, and user configurability.
#
# Users can:
#   - Provide their own data (x_train, u_train, x_test, u_test)
#   - Specify domain boundaries, number of collocation points, and expected frequencies
#   - The code will encode the data, train the PINN, and plot results
#   - An example with synthetic data is provided at the end
#
# =============================================================================

import torch
import torch.autograd as autograd
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from pyDOE import lhs # Latin Cube Hypersampling
import rff
from pathlib import Path

# ------------------- Reproducibility and Determinism -------------------
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

# ------------------- Precision Control (float32/float64) ---------------
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)

# ------------------- Device Selection ----------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =============================================================================
# PINN Model Definition
# =============================================================================
def init_weights(m):
    """Xavier initialization for Linear layers."""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)

class PINN(nn.Module):
    """
    Physics-Informed Neural Network with Fourier Embedding.
    """
    def __init__(self, hidden_units, input_size, fourier_encoder_2, fourier_encoder_50):
        super().__init__()
        self.loss_function = nn.MSELoss(reduction='mean')
        self.input_size = input_size
        self.fourier_encoder_2 = fourier_encoder_2
        self.fourier_encoder_50 = fourier_encoder_50
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=hidden_units),
            nn.Tanh(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.Tanh(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.Tanh(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.Tanh(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.Tanh(),
            nn.Linear(in_features=hidden_units, out_features=1),
        )

    def lossTrain(self, x_train, u_train):
        loss_train = self.loss_function(self.forward(x_train), u_train)
        return loss_train

    def lossBC(self, x_bc, u_bc):
        loss_bc = self.loss_function(self.forward(x_bc), u_bc)
        return loss_bc

    def lossPDE(self, x_PDE):
        # Ensure all tensors are on the same device and dtype as x_PDE
        x = x_PDE.clone().detach().requires_grad_(True)
        x = x.to(device=x_PDE.device, dtype=x_PDE.dtype)
        u = self.forward(x)
        ones_like_u = torch.ones_like(u).to(x.device, dtype=u.dtype)
        u_x = autograd.grad(u, x, ones_like_u, retain_graph=True, create_graph=True)[0]
        ones_like_ux = torch.ones_like(u_x).to(x.device, dtype=u_x.dtype)
        u_xx = autograd.grad(u_x, x, ones_like_ux, create_graph=True)[0]
        # The PDE below is for the example problem. For custom problems, modify as needed.
        f = u_xx + torch.mul(torch.sin(2*np.pi*x), 4*np.pi**2) + torch.mul(torch.sin(50*np.pi*x), 250*np.pi**2)
        zeros_target = torch.zeros(f.shape[0], 1, device=x.device, dtype=f.dtype)
        return self.loss_function(f, zeros_target)

    def loss(self, x_train, u_train, x_bc, u_bc, x_PDE, weights="static"):
        loss_pde = self.lossPDE(x_PDE)
        loss_bc = self.lossBC(x_bc, u_bc)
        loss_train = self.lossTrain(x_train, u_train)
        if weights == "static":
            weight_pde = 1e-2
            weight_bc = 1e6
            weight_train = 1
            total = loss_pde * weight_pde + loss_train * weight_train + loss_bc * weight_bc
            return loss_train * weight_train, loss_bc * weight_bc, loss_pde * weight_pde, total
        elif weights == "adaptive simple":
            s = loss_pde + loss_train + loss_bc
            weight_pde = loss_pde / s
            weight_bc = loss_bc / s
            weight_train = loss_train / s
            total = loss_pde * weight_pde + loss_train * weight_train + loss_bc * weight_bc
            return loss_train * weight_train, loss_bc * weight_bc, loss_pde * weight_pde, total

    def forward(self, x: torch.Tensor):
        if x.shape[1] == self.input_size:
            return self.layer_stack(x)
        else:
            # Always encode on the SAME device as the input (not CPU), to avoid device mismatch in rff
            x_50 = self.fourier_encoder_50(x)
            x_2 = self.fourier_encoder_2(x)
            X = torch.hstack((x_2, x_50))
            return self.layer_stack(X)

# =============================================================================
# Utility Functions
# =============================================================================


# ------------------- Global Fourier Bases and Encoders -------------------
# Create the global Fourier encoders 
FREQUENCIES = [2, 50]
ENCODED_SIZE = 2

# Always use torch.float and CPU for bases
B_2 = 2 * torch.randn(ENCODED_SIZE, 1, device='cpu', dtype=torch.float)
B_50 = 50 * torch.randn(ENCODED_SIZE, 1, device='cpu', dtype=torch.float)
FOURIER_ENCODER_2 = rff.layers.GaussianEncoding(b=B_2.cpu())
FOURIER_ENCODER_50 = rff.layers.GaussianEncoding(b=B_50.cpu())
FOURIER_ENCODERS = [FOURIER_ENCODER_2, FOURIER_ENCODER_50]


def encode_data(x: torch.Tensor, encoders: list, device: str, dtype: torch.dtype) -> torch.Tensor:
    """
    Encode input x using a list of Fourier encoders and concatenate the results.
    Args:
        x: Input tensor of shape (N, 1)
        encoders: List of rff.layers.GaussianEncoding objects
        device: Target device (e.g., 'cpu' or 'cuda')
        dtype: Target torch dtype
    Returns:
        Encoded tensor of shape (N, 2 * encoded_size * len(encoders))
    """
    encoded = [enc(x.cpu()).to(device=device, dtype=dtype) for enc in encoders]
    return torch.hstack(encoded)

# =============================================================================
# Training Function
# =============================================================================
def train_PINN(
    x_train: torch.Tensor,
    u_train: torch.Tensor,
    x_test: torch.Tensor,
    u_test: torch.Tensor,
    x_bc: torch.Tensor,
    u_bc: torch.Tensor,
    x_PDE: torch.Tensor,
    fourier_encoders: list,
    hidden_units: int = 100,
    lr: float = 1e-4,
    decay_rate: float = 0.96,
    epochs_before_decay: int = 10000,
    epochs: int = 50000,
    weights: str = "static",
    save_path: str = "models/pinn.pth"
) -> tuple:
    """
    Train a PINN with Fourier embedding on the provided data.
    Args:
        x_train, u_train: Training data and targets
        x_test, u_test: Test data and targets
        x_bc, u_bc: Boundary condition data and targets
        x_PDE: Collocation points for PDE loss
        fourier_encoders: List of Fourier encoders
        hidden_units: Number of hidden units in each layer
        lr: Learning rate
        decay_rate: Exponential decay rate for learning rate
        epochs_before_decay: Epochs before each decay
        epochs: Total number of epochs
        weights: Loss weighting scheme
        save_path: Path to save the trained model
    Returns:
        model: Trained PINN model
        history: Dictionary with loss curves and predictions
    """
    encoded_size = fourier_encoders[0].b.shape[0]
    input_size = 2 * encoded_size * len(fourier_encoders)
    fourier_encoder_2 = fourier_encoders[0]
    fourier_encoder_50 = fourier_encoders[1]

    # Encode data as in the original code: encode on CPU, then move to device/DTYPE
    x_train_50 = fourier_encoder_50(x_train.cpu()).to(device=device, dtype=DTYPE)
    x_train_2 = fourier_encoder_2(x_train.cpu()).to(device=device, dtype=DTYPE)
    x_train_encoded = torch.hstack((x_train_2, x_train_50))

    x_test_50 = fourier_encoder_50(x_test.cpu()).to(device=device, dtype=DTYPE)
    x_test_2 = fourier_encoder_2(x_test.cpu()).to(device=device, dtype=DTYPE)
    x_test_encoded = torch.hstack((x_test_2, x_test_50))

    x_bc_50 = fourier_encoder_50(x_bc.cpu()).to(device=device, dtype=DTYPE)
    x_bc_2 = fourier_encoder_2(x_bc.cpu()).to(device=device, dtype=DTYPE)
    x_bc_encoded = torch.hstack((x_bc_2, x_bc_50))

    # Model
    model = PINN(hidden_units=hidden_units, input_size=input_size, fourier_encoder_2=fourier_encoder_2, fourier_encoder_50=fourier_encoder_50)
    model.apply(init_weights)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=False)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_rate)

    PDE_loss_values = []
    train_loss_values = []
    total_loss_values = []
    test_loss_values = []
    BC_loss_values = []
    u_evolution = torch.zeros(x_test_encoded.shape[0], 1, device=device, dtype=DTYPE)
    epoch_count = []

    for epoch in range(epochs):
        epoch_count.append(epoch)
        model.train()
        train_loss, BC_loss, PDE_loss, total_loss = model.loss(x_train_encoded, u_train, x_bc_encoded, u_bc, x_PDE, weights=weights)
        PDE_loss_values.append(PDE_loss.item())
        total_loss_values.append(total_loss.item())
        train_loss_values.append(train_loss.item())
        BC_loss_values.append(BC_loss.item())
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        # Optional: Gradient clipping (uncomment to activate)
        #clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch % epochs_before_decay == 0:
            lr_scheduler.step()
        if epoch % 500 == 0:
            model.eval()
            with torch.inference_mode():
                u_pred = model.forward(x_test_encoded)
                u_evolution = torch.hstack((u_evolution, u_pred.to(device=u_evolution.device, dtype=u_evolution.dtype)))
                test_loss = model.loss_function(u_pred, u_test)
            test_loss_values.append(test_loss)
            print(f"Epoch: {epoch} | PDE loss: {PDE_loss:.5f} | Train loss: {train_loss:.5f} | BC loss: {BC_loss:.5f} | Total loss: {total_loss:.5f}\n")
    print(f"Epoch: {epoch_count[-1]+1} | PDE loss: {PDE_loss_values[-1]:.5f} | Train loss: {train_loss_values[-1]:.5f} | BC loss: {BC_loss_values[-1]:.5f} | Total loss: {total_loss_values[-1]:.5f}\n")

    # Save model
    model_dir = Path(save_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving model to: {save_path}")
    torch.save(obj=model.state_dict(), f=save_path)

    return model, {
        'PDE_loss_values': PDE_loss_values,
        'train_loss_values': train_loss_values,
        'total_loss_values': total_loss_values,
        'test_loss_values': test_loss_values,
        'BC_loss_values': BC_loss_values,
        'u_evolution': u_evolution,
        'epoch_count': epoch_count,
        'x_test_encoded': x_test_encoded
    }

# =============================================================================
# Plotting Functions
# =============================================================================
def plot_results(x_train, u_train, x_test, u_test, u_hat, u_evolution, epoch_count, BC_loss_values, PDE_loss_values, total_loss_values, train_loss_values):
    """Plot predictions, error, and loss curves."""
    u_hat = u_hat.detach().cpu()
    u_test = u_test.detach().cpu()
    u_train = u_train.detach().cpu()
    error = np.subtract(u_test, u_hat)

    fig_1 = plt.figure(10)
    plt.plot(x_train.detach().cpu(), u_train, label="Noisy data", color='red')
    plt.plot(x_test.detach().cpu(), u_test, label="Exact", color='orange')
    plt.plot(x_test.detach().cpu(), u_hat, label="Final estimate", linestyle='dashed', linewidth=1.2, color='k')
    plt.title("u(x)")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.show()
    fig_1.savefig('images/prediction')

    fig_2 = plt.figure(20)
    plt.plot(x_test.detach().cpu(), error)
    plt.title("Point_wise_error")
    plt.xlabel("x")
    plt.ylabel("error")
    plt.ticklabel_format(style='sci', axis='y')
    plt.show()
    fig_2.savefig('images/error')

    fig_3 = plt.figure(100)
    plt.semilogy(epoch_count, BC_loss_values, label="BC loss")
    plt.semilogy(epoch_count, PDE_loss_values, label="PDE loss")
    plt.semilogy(epoch_count, total_loss_values, label="total loss")
    plt.semilogy(epoch_count, train_loss_values, label="train loss")
    plt.title("Loss functions")
    plt.xlabel("epochs")
    plt.ylabel("losses")
    plt.legend()
    plt.show()
    fig_3.savefig('images/loss_functions')

# =============================================================================
# Example Usage: Synthetic Data (Default)
# =============================================================================
def ground_truth(x: torch.Tensor) -> torch.Tensor:
    """Ground truth function for synthetic data."""
    return torch.sin(2*np.pi*x) + torch.mul(torch.sin(50*np.pi*x), 0.1)

def prepare_synthetic_data(
    x_min: float = 0.0,
    x_max: float = 1.0,
    N_train: int = 50,
    N_test: int = 200,
    Nf: int = 1000,
    device: str = device,
    dtype: torch.dtype = DTYPE
) -> tuple:
    """
    Prepare synthetic data for the PINN example.
    Returns:
        x_train, u_train, x_test, u_test, x_bc, u_bc, x_PDE
    """
    x_train = torch.linspace(x_min, x_max, N_train).view(-1,1)
    x_test = torch.linspace(x_min, x_max, N_test).view(-1,1)
    x_PDE = torch.from_numpy(x_min + (x_max - x_min) * lhs(1, Nf)).type(torch.float).to(device)
    x_bc = torch.Tensor([x_min, x_max]).view(-1,1).type(torch.float)

    # Move to device
    x_train = x_train.to(device)
    x_test = x_test.to(device)
    x_bc = x_bc.to(device)

    # Generate targets
    u_train = ground_truth(x_train)
    u_test = ground_truth(x_test)
    u_bc = torch.Tensor([0, 0]).view(-1, 1).type(torch.float).to(device)

    # Add noise to training data
    noise = torch.randn(u_train.shape[0], 1, device=device)
    u_train = u_train + noise

    # Move all input data to device and dtype BEFORE encoding
    x_train = x_train.to(device=device, dtype=dtype)
    x_test = x_test.to(device=device, dtype=dtype)
    x_bc = x_bc.to(device=device, dtype=dtype)
    x_PDE = x_PDE.to(device=device, dtype=dtype)
    u_train = u_train.to(device=device, dtype=dtype)
    u_test = u_test.to(device=device, dtype=dtype)
    u_bc = u_bc.to(device=device, dtype=dtype)

    return x_train, u_train, x_test, u_test, x_bc, u_bc, x_PDE



if __name__ == "__main__":
    # Prepare synthetic data
    x_train, u_train, x_test, u_test, x_bc, u_bc, x_PDE = prepare_synthetic_data()

    # Use the global Fourier encoders (created after seeding, before any data/noise/model)
    fourier_encoders = FOURIER_ENCODERS

    # ---------------------- Train the PINN ----------------------
    model, history = train_PINN(
        x_train, u_train, x_test, u_test, x_bc, u_bc, x_PDE,
        fourier_encoders=fourier_encoders,
        hidden_units=100,
        lr=1e-4,
        decay_rate=0.96,
        epochs_before_decay=10000,
        epochs=50000,
        weights="static",
        save_path="models/pinn.pth"
    )

    # ---------------------- Plot Results ----------------------
    u_hat = model.forward(history['x_test_encoded'].to(device))

    # Plot true u(x) for comparison
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x_test.cpu().numpy(), ground_truth(x_test).cpu().numpy(), label="True u(x)", color="orange", linewidth=2)
    plt.plot(x_test.cpu().numpy(), u_hat.cpu().numpy(), label="PINN prediction", color="k", linestyle="dashed")
    plt.scatter(x_train.cpu().numpy(), u_train.cpu().numpy(), label="Noisy train data", color="red", s=10)
    plt.title("PINN vs True Solution")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.show()

    plot_results(
        x_train, u_train, x_test, u_test, u_hat, history['u_evolution'],
        history['epoch_count'], history['BC_loss_values'], history['PDE_loss_values'],
        history['total_loss_values'], history['train_loss_values']
    )

