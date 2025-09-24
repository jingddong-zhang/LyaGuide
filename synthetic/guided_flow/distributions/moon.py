from .base import BaseDistribution, EPS
import torch
import numpy as np
import sklearn
import os
import torch.nn as nn
from tqdm import tqdm

# Define the equivalent PyTorch model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)
    
    def train(self, x, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        print(f"Training MLP with {x.shape[0]} samples")
        for _ in tqdm(range(1000)):
            optimizer.zero_grad()
            pred = self(x)
            loss = nn.CrossEntropyLoss()(pred, y)
            loss.backward()
            optimizer.step()
        return self


class MoonDistribution(BaseDistribution):
    def __init__(self, noise: float = 0.05):
        super().__init__()
        self.noise = noise
        self.classifier = self.get_classifier()

    def get_classifier(self):
        try:
            # load cached classifier
            os.makedirs('logs/distributions', exist_ok=True)
            classifier = MLP(2, [64, 64], 2)
            classifier.load_state_dict(torch.load('logs/distributions/moon_classifier.pth'))
        except Exception as e:
            print("Training new classifier.", e)
            # train new classifier
            x, y = sklearn.datasets.make_moons(n_samples=1000, noise=0)
            classifier = MLP(2, [64, 64], 2).train(
                (torch.tensor(x, dtype=torch.float32) + torch.tensor([-0.5, -0.25], dtype=torch.float32)) / 1.5, 
                torch.tensor(y, dtype=torch.long)
            )
            torch.save(classifier.state_dict(), 'logs/distributions/moon_classifier.pth')
        return classifier

    def sample(self, batch_size: int, device: str = 'cuda') -> torch.Tensor:
        x, _ = sklearn.datasets.make_moons(n_samples=batch_size, noise=self.noise) # (B, 2)
        x = torch.tensor(x, device=device, dtype=torch.float32) + torch.tensor([-0.5, -0.25], device=device, dtype=torch.float32)
        return x / 1.5
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Probability for moons is complicated, not implemented yet")
    
    def prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Probability for moons is complicated, not implemented yet")

    # def get_J(self, x) -> torch.Tensor:
    #     # I want upper half of the moon to have J=1, and lower half to have J=0.
    #     label = self.classifier(x.cpu()) # (B,)
    #     return label.to(x.device)[:, 1] * 3

    # def get_J(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     Energy function for moon dataset:
    #     assign energy based on distance to midpoints of the two branches.
        
    #     Args:
    #         x: (B, 2) tensor of samples
    #     Returns:
    #         J: (B,) energy values
    #     """
    #     # 假设上半月的中点和下半月的中点
    #     centers = torch.tensor([
    #         [0.5, 0.75],   # 上半月大概中心
    #         [0.5, -0.25],  # 下半月大概中心
    #     ]).float().to(x.device)

    #     # 计算到每个中心的距离 (B, 2)
    #     dists = torch.cdist(x, centers)  
        
    #     # 取最小值 (B,)
    #     output = dists.min(dim=1).values  
        
    #     assert output.shape == x.shape[:-1]
    #     return output
    

    def get_J(self, x1):
        """
        Args: x1 (B, 2)
        Returns: J (B,)
        """
        centers = torch.tensor([[0,0]]).float().to(x1.device)
        dists = (torch.cdist(x1, centers))  # (B, n_centers)
        output = dists.min(dim=1).values
        assert output.shape == x1.shape[:-1]
        return output # 与最近中心的距离
    
    def get_J_x(self, x1):
        """
        Args:
            x1: Tensor, shape (B, 2)
        Returns:
            grad_J: Tensor, shape (B, 2), i.e., ∇_x J(x1)
        Notes:
            J(x) = min_k tanh( ||x - c_k||_2 )
            ∇J(x) = sech^2(d_{k*}) * (x - c_{k*}) / d_{k*}, where k* = argmin_k tanh(d_k)
        """
        # centers: (K, 2)\
        eps = 1e-8
        centers = torch.tensor([[0,0]], 
                            device=x1.device, dtype=x1.dtype)

        # diffs: (B, K, 2), d: (B, K)
        diffs = x1.unsqueeze(1) - centers.unsqueeze(0)
        d = torch.linalg.norm(diffs, dim=-1).clamp_min(eps)   # avoid division by zero

        # s = tanh(d) and choose the minimizing center
        s = torch.tanh(d)                                     # (B, K)
        idx = torch.argmin(s, dim=1)                          # (B,)

        # gather k* terms
        bidx = torch.arange(x1.size(0), device=x1.device)
        d_star    = d[bidx, idx].unsqueeze(1)                 # (B, 1)
        diff_star = diffs[bidx, idx, :]                       # (B, 2)

        # sech^2(d) = 1 - tanh^2(d)
        sech2 = 1.0 - torch.tanh(d_star)**2                   # (B, 1)

        # ∇J = sech^2(d_{k*}) * (x - c_{k*}) / d_{k*}
        grad_J = sech2 * (diff_star / d_star)                 # (B, 2)
        assert grad_J.shape == x1.shape
        return grad_J

    # def get_J(self, x):
    #      # I want upper half of the moon to have J=1, and lower half to have J=0.
    #     label = self.classifier(x.cpu()) # (B,)
    #     return label.to(x.device)[:, 1] * 3 

    def get_J_asym(self,x) -> torch.Tensor:
    #     # I want upper half of the moon to have J=1, and lower half to have J=0.
        label = self.classifier(x.cpu()) # (B,)
        return label.to(x.device)[:, 1] * 3
    
    def get_J_asym_x(self,x) -> torch.Tensor:
        return torch.zeros_like(x)
    def __name__(self):
        return f'Moons'
