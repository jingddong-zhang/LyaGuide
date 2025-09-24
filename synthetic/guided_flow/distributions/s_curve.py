from .base import BaseDistribution, EPS
import torch
import numpy as np
import sklearn
import math
class SCurveDistribution(BaseDistribution):
    def __init__(self, noise: float = 0.05):
        super().__init__()
        self.noise = noise

    def sample(self, batch_size: int, device: str = 'cuda') -> torch.Tensor:
        X, _ = sklearn.datasets.make_s_curve(n_samples=batch_size, noise=self.noise) # (B, 2)
        x = torch.tensor(X[:, [0]], device=device, dtype=torch.float32)
        y = torch.tensor(X[:, [2]], device=device, dtype=torch.float32)
        return torch.cat([x, y / 2], dim=-1)
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Probability for s-curve is complicated, not implemented yet")
    
    def prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Probability for s-curve is complicated, not implemented yet")

    def get_J(self, x) -> torch.Tensor:
        """
        Allows grad.
        """
        return torch.tanh((x[:, 0] - x[:, 1])**2/2)#**2/5 #* 5
    def get_J_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        解析梯度 ∇J(x)，对应 J(x) = tanh((x0 - x1)^2 / 2)
        Args:
            x: (B, 2) tensor
        Returns:
            grad: (B, 2) tensor
        """
        d = x[:, 0] - x[:, 1]             # (B,)
        u = 0.5 * d**2                    # (B,)

        # sech^2(u) = 1 / cosh^2(u)
        sech2 = 1.0 / torch.cosh(u).pow(2)

        grad_x0 = sech2 * d
        grad_x1 = -sech2 * d

        grad = torch.stack([grad_x0, grad_x1], dim=1)  # (B, 2)
        return grad
    # def get_J_x(self, x, eps: float = 1e-12) -> torch.Tensor:
    #     """
    #     解析梯度 ∇J(x)，对应 J(x) = 5 * |x0 - x1|
    #     Args:
    #         x: (B, 2) tensor
    #         eps: small number to avoid undefined gradient at 0
    #     Returns:
    #         grad: (B, 2) tensor
    #     """
    #     d = x[:, 0] - x[:, 1]             # (B,)
    #     # sign_d = torch.sign(d)

    #     # # 在 d=0 时强制设为 0（次梯度）
    #     # sign_d = torch.where(d.abs() < eps, torch.zeros_like(d), sign_d)

    #     grad_x0 =  d*2/5
    #     grad_x1 = -d*2/5
    #     grad = torch.stack([grad_x0, grad_x1], dim=1)  # (B, 2)
    #     return grad
    
    # def get_J_asym(self, x) -> torch.Tensor:
    #     """
    #     Allows grad.
    #     """
    #     return torch.tanh((x[:, 0] + x[:, 1])**2/10)

    def get_J_asym(self, x1):
        """
        Args: x1 (B, 2)
        Returns: J (B,)
        """
        centers = torch.tensor([ [-1/math.sqrt(2),1/math.sqrt(2)], [1/math.sqrt(2),-1/math.sqrt(2)]]).float().to(x1.device)
        dists = (torch.cdist(x1, centers))*2  # (B, n_centers)
        output = dists.min(dim=1).values
        assert output.shape == x1.shape[:-1]
        return output # 与最近中心的距离
        # return (x1[:, 0] + x1[:, 1]).abs()*5
    
    def get_J_asym_x(self, x1):
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
        centers = torch.tensor([ [-1/math.sqrt(2),1/math.sqrt(2)], [1/math.sqrt(2),-1/math.sqrt(2)]]).float().to(x1.device)

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
    


    # def get_J_asym_x(self, x, eps: float = 1e-12) -> torch.Tensor:
    #     d = x[:, 0] + x[:, 1]             # (B,)
    #     # sign_d = torch.sign(d)

    #     # 在 d=0 时强制设为 0（次梯度）
    #     # sign_d = torch.where(d.abs() < eps, torch.zeros_like(d), sign_d)

    #     grad_x0 = 5 * 2* d
    #     grad_x1 = 5 * 2* d
    #     grad = torch.stack([grad_x0, grad_x1], dim=1)  # (B, 2)
    #     return grad

    # def get_J_asym_x(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     解析梯度 ∇J(x)，对应 J(x) = tanh((x0 - x1)^2 / 2)
    #     Args:
    #         x: (B, 2) tensor
    #     Returns:
    #         grad: (B, 2) tensor
    #     """
    #     d = x[:, 0] + x[:, 1]             # (B,)
    #     u = d**2/10                    # (B,)

    #     # sech^2(u) = 1 / cosh^2(u)
    #     sech2 = 1.0 / torch.cosh(u).pow(2)

    #     grad_x0 = sech2 * d
    #     grad_x1 = sech2 * d

    #     grad = torch.stack([grad_x0, grad_x1], dim=1)  # (B, 2)
    #     return grad
    
    def __name__(self):
        return f'S-Curve'
