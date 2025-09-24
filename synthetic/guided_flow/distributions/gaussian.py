from .base import BaseDistribution
import torch
import math

class GaussianDistribution(BaseDistribution):
    def __init__(
        self, 
        mean: torch.Tensor = torch.zeros(2), 
        std: torch.Tensor = torch.ones(2)
    ):
        super().__init__()
        self.mean = mean
        self.std = std

    def sample(self, batch_size: int, device: str = 'cuda') -> torch.Tensor:
        eps = torch.randn(batch_size, 2, device=device)
        return eps * self.std.to(device) + self.mean.to(device)
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * ((x - self.mean.to(x.device)) / self.std.to(x.device)).pow(2).sum(-1) - torch.log(self.std.to(x.device)).sum(-1) - 0.5 * math.log(2 * torch.pi)

    def get_J(self) -> torch.Tensor:
        raise NotImplementedError("J for gaussian is not implemented yet")

    def get_J_x(self) -> torch.Tensor:
        raise NotImplementedError("Jx for gaussian is not implemented yet")
    
    def get_J_asym(self) -> torch.Tensor:
        raise NotImplementedError("Jx for gaussian is not implemented yet")

    def __name__(self):
        return f'Gaussian'

class EightGaussiansDistribution(BaseDistribution):
    def __init__(
        self, 
        centers: torch.Tensor = torch.tensor(
            [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1.0 / math.sqrt(2), 1.0 / math.sqrt(2)),
                (1.0 / math.sqrt(2), -1.0 / math.sqrt(2)),
                (-1.0 / math.sqrt(2), 1.0 / math.sqrt(2)),
                (-1.0 / math.sqrt(2), -1.0 / math.sqrt(2)),
            ], dtype=torch.float32
        ), 
        std: torch.Tensor = torch.ones(8, 2) * 0.1
    ):
        super().__init__()
        self.centers = centers
        self.std = std

    def sample(self, batch_size: int, device: str = 'cuda') -> torch.Tensor:
        multi = torch.multinomial(torch.ones(8, device=device), batch_size, replacement=True) # size (batch_size,)
        eps = torch.randn(batch_size, 2, device=device)
        centers = self.centers.to(device).repeat(batch_size, 1, 1)[torch.arange(batch_size, device=device), multi]
        stds = self.std.to(device).repeat(batch_size, 1, 1)[torch.arange(batch_size, device=device), multi]
        return centers + eps * stds
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.prob(x).clamp(1e-10).log()

    def prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Approximate with the PDF of 8 non-overlapping gaussians.
        """
        one_log_prob = -0.5 * ((x.unsqueeze(1) - self.centers.to(x.device).unsqueeze(0)) / self.std.to(x.device).unsqueeze(0)).pow(2).sum(-1) # (B, 8, 2) -> (B, 8)
        one_log_prob_z = torch.log(self.std.to(x.device)).sum(-1) - 0.5 * math.log(2 * torch.pi) # (B, 8)
        prob = (torch.exp(one_log_prob) / torch.exp(one_log_prob_z)).mean(-1)
        return prob

    # def get_J_asym(self, x1: torch.Tensor) -> torch.Tensor:
    #     """
    #     Args: x1 (B, 2)
    #     Returns: J (B,)
    #     """
    #     # theta in [-pi, pi]
    #     theta = torch.atan2(x1[..., 0], x1[..., 1])
    #     theta_out = torch.zeros_like(theta)
    #     intervals = torch.linspace(-torch.pi, torch.pi, 9) + torch.pi / 8
    #     for i in range(len(intervals) - 1):
    #         mask = (theta >= intervals[i]) & (theta < intervals[i+1])
    #         theta_out[mask] = i
    #     theta_out[theta < intervals[0]] = len(intervals) - 2

    #     assert theta_out.shape == x1.shape[:-1]
    #     return theta_out    

    def get_J_asym(self, x1: torch.Tensor, discrete=False,k: float = 0.1) -> torch.Tensor:
        """
        分段线性递增函数
        Args:
            x1: (B, 2)
            k: 斜率
        Returns:
            J: (B,)
        """
        # theta ∈ [-pi, pi]
        theta = torch.atan2(x1[..., 0], x1[..., 1])  # (B,)
        
        # 区间划分（8段，偏移 π/8）
        intervals = torch.linspace(-torch.pi, torch.pi, 9, device=x1.device) + torch.pi / 8
        centers = torch.tensor([[-1.0 / math.sqrt(2), -1.0 / math.sqrt(2)],[-1,0],
                [-1.0 / math.sqrt(2), 1.0 / math.sqrt(2)], [0,1], 
                [1.0 / math.sqrt(2), 1.0 / math.sqrt(2)],[1,0],
                [1.0 / math.sqrt(2), -1.0 / math.sqrt(2)], [0,-1]]).float().to(x1.device)
        theta2 = torch.atan2(centers[..., 0], centers[..., 1])  # (B,)
        if not discrete: 
            
            dists = torch.tanh(torch.cdist(x1, centers))  # (B, n_centers)
            output = dists.min(dim=1).values
            assert output.shape == x1.shape[:-1]
         
            theta_out = torch.zeros_like(theta)
            for i in range(len(intervals) - 1):
                mask = (theta >= intervals[i]) & (theta < intervals[i+1])
                mask2 = (theta2 >= intervals[i]) & (theta2 < intervals[i+1])
                theta_out[mask] = torch.tanh(torch.cdist(x1[mask], centers[mask2])[:,0]) 
            theta_out[theta < intervals[0]] = len(intervals) - 2
            assert theta_out.shape == x1.shape[:-1]
            return theta_out 

            # 输出初始化
            # J = torch.zeros_like(theta)
            # # 遍历每个区间
            # base_val = 0.0  # 每段起点的基准值
            # for i in range(len(intervals) - 1):
            #     left, right = intervals[i], intervals[i+1]
            #     mask = (theta >= left) & (theta < right)
            #     J[mask] = base_val + k * (theta[mask] - left) 
            #     # 更新下一个区间的起点值
            #     base_val +=  (right - left)
            
            # # 处理 theta < intervals[0] 的点（周期性 wrap-around）
            # mask = theta < intervals[0]
            # if mask.any():
            #     left, right = intervals[0] - 2*torch.pi, intervals[1] - 2*torch.pi
            #     J[mask] = base_val + k * (theta[mask] - left) 
            # return J
        else:
            theta_out = torch.zeros_like(theta)
            for i in range(len(intervals) - 1):
                mask = (theta >= intervals[i]) & (theta < intervals[i+1])
                theta_out[mask] = i
            theta_out[theta < intervals[0]] = len(intervals) - 2
            assert theta_out.shape == x1.shape[:-1]
            return theta_out    

    def get_J_asym_x(self, x1: torch.Tensor, k: float = 0.1, eps: float = 1e-12) -> torch.Tensor:
        """
        解析梯度：∇_x J_asym_linear(x)
        与 get_J_asym_linear 相对应，但这里不需要区间判断，
        因为每段的 dJ/dθ = k 常数（分界点除外，零测集）。

        Args:
            x1: (B, 2), 其中 x1[...,0] = x, x1[...,1] = y
            k:  每段线性函数的斜率
            eps: 防止除零的稳定项
        Returns:
            grad_J: (B, 2)
        """
        x = x1[..., 0]
        y = x1[..., 1]
        r2 = (x * x + y * y).clamp_min(eps)  # x^2 + y^2

        # 对 θ = atan2(x, y) 的梯度：
        dtheta_dx = y / r2
        dtheta_dy = -x / r2

        # ∇J = k * ∇θ
        grad_J_x = k * dtheta_dx
        grad_J_y = k * dtheta_dy

        grad_J = torch.stack([grad_J_x, grad_J_y], dim=-1)  # (B, 2)
        return grad_J


    # def get_J_asym(self, x1: torch.Tensor) -> torch.Tensor:
    #     """
    #     Args: x1 (B, 2)
    #     Returns: J (B,)
    #     """
    #     # theta in [-pi, pi]
    #     theta = torch.atan2(x1[..., 0], x1[..., 1])
    #     theta_out = torch.zeros_like(theta)
    #     intervals = torch.linspace(-torch.pi, torch.pi, 9) + torch.pi / 8
    #     for i in range(len(intervals) - 1):
    #         mask = (theta >= intervals[i]) & (theta < intervals[i+1])
    #         theta_out[mask] = i
    #     theta_out[theta < intervals[0]] = len(intervals) - 2

    #     assert theta_out.shape == x1.shape[:-1]
    #     return theta_out    

    

    def get_J(self, x1):
        """
        Args: x1 (B, 2)
        Returns: J (B,)
        """
        centers = torch.tensor([[1,0], [-1,0], [0,1], [0,-1]]).float().to(x1.device)
        dists = torch.tanh(torch.cdist(x1, centers))  # (B, n_centers)
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
        centers = torch.tensor([[1,0], [-1,0], [0,1], [0,-1]], 
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
    
    def __name__(self):
        return f'8 Gaussians'
