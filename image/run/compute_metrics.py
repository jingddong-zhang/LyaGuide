import os
import tyro

from gflow_img.config.inverse_problem import InverseProblemConfig
from gflow_img.inverse.inverse_problems import InverseProblem
from gflow_img.inverse.metrics import compute_metrics
from gflow_img.utils.misc import deterministic, set_cuda_visible_device

def compute_save_metrics(cfg):
    print(os.path.join(cfg.make_path()))
    fid_score, lpip_score, psnr_score, ssim_score = compute_metrics(
        cfg, 
        os.path.join(cfg.make_path(), 'x_gen'), os.path.join(cfg.make_path(), 'x_gt')
    )

    with open(os.path.join(cfg.make_path(), f'metrics.txt'), 'a+') as f:
        f.write(f"FID Score: {fid_score.item()}\n")
        f.write(f"LPIPS Score: {lpip_score.item()}\n")
        f.write(f"PSNR Score: {psnr_score.item()}\n")
        f.write(f"SSIM Score: {ssim_score.item()}\n")


if __name__ == '__main__':
    cfg = tyro.cli(InverseProblemConfig)
    cfg.problem = "deblurring"
    cfg.guide_method = "nabla_x1_J_x1"
    cfg.schedule = "cosine_decay"
    # cfg.schedule = "const"
    cfg.start_time = "0.0"
    cfg.schedule_ratio = "False"
    cfg.guide_scale = "10.0"
    cfg.time_ratio_eps = "0.01"
    cfg.clamp_x = "False"
    cfg.lya = "True"
    deterministic(cfg.seed)
    set_cuda_visible_device(cfg)

    compute_save_metrics(cfg)
