import os
import torch
import tqdm
from guided_flow.config.guidance_training import GuidanceTrainingConfig
from guided_flow.distributions.base import get_distribution
from guided_flow.backbone.mlp import MLP, EnergyMLP
from guided_flow.flow.conditional_flow_matching import get_cfm
from guided_flow.utils.misc import deterministic, save_config, set_cuda_visible_device
import torch.nn.functional as F
import tyro
# from guided_flow.utils.kl_divergence import compute_kl_divergence
MLP_WIDTH = 256
TRAINING_B = 256 # OT CFM training batch size

# ========= 1) 训练 V_theta：用 few-shot (x1, J(x1)) 回归能量 =========
def train_V(
    x1_sampler,
    J_true,                # callable: J_true(x1) -> (B,)
    width=128,
    device='cuda',
    n_support=1024,        # few-shot 样本数
    batch_size=256,
    lr=1e-3,
    num_steps=2000,
    use_huber=True,        # 是否用 Huber 更稳
    grad_reg=0.0,          # 对 ∇V 做 L2 正则（0 表示不用）
):
    V_net = EnergyMLP(dim=2, out_dim=1, w=width, time_varying=False).to(device)
    opt = torch.optim.AdamW(V_net.parameters(), lr=lr)

    # 采样 few-shot 支持集
    with torch.no_grad():
        Xs = x1_sampler(batch_size=n_support).to(device)          # (N, 2)
        ys = J_true(Xs).float().view(-1, 1)                        # (N, 1)

    ds = torch.utils.data.TensorDataset(Xs, ys)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    huber = torch.nn.SmoothL1Loss(beta=1.0,reduction='none')  # Huber
    mse   = torch.nn.MSELoss()

    V_net.train()
    for _ in tqdm.tqdm(range(num_steps), desc='train_V'):
        for xb, yb in dl:
            xb = xb.to(device); yb = yb[:,0].to(device)
            xb.requires_grad_(grad_reg > 0.0)

            pred = V_net(xb)                 # (B,1)
            alpha = 1.0   # 调节强度
            w = torch.softmax(-alpha * yb.view(-1), dim=0)   # (B,)

            if use_huber:
                loss = huber(pred, yb)
                loss = (loss*w).mean()
                # loss = loss.mean()
            else:
                loss = mse(pred, yb)

            # 可选的梯度正则：鼓励 V 平滑，避免噪声 few-shot 过拟合
            if grad_reg > 0.0:
                grad = torch.autograd.grad(pred.sum(), xb, create_graph=True, retain_graph=True)[0]  # (B,2)
                loss = loss + grad_reg * (grad.pow(2).sum(dim=1).mean())

            opt.zero_grad()
            loss.backward()
            opt.step()

    return V_net,Xs,ys


# ========= 2) 训练 g_phi：用 -k ∇V_theta(x1_pred) 作为“理想 guidance” =========
def grad_V_of(V_net, x):
    with torch.enable_grad():
        x = x.requires_grad_(True)
        V = V_net(x).sum()                 # V_net: (B,2)->(B,1)
        g = torch.autograd.grad(V, x, create_graph=False, retain_graph=False)[0]
    return g                                # (B,2)

# --------- Lyapunov-inequality 的损失：softplus( <∇V, u+U> + δ V ) ---------
def lya_hinge_loss(gradV, Vval, u_base, U_res, delta=0.1, smooth='softplus'):
    """
    gradV: (B,2), Vval: (B,1) or (B,), u_base/U_res: (B,2)
    returns scalar loss
    """
    ip = (gradV * (u_base + U_res)).sum(dim=1)      # <∇V, u+U>  (B,)
    Vb = Vval.view(-1)                               # (B,)
    viol = ip + delta * Vb                           # violation: should be <= 0
    if smooth == 'softplus':
        return F.softplus(viol, beta=10.0).mean()    # 平滑 hinge，beta 可调
    elif smooth == 'relu':
        return F.relu(viol).mean()
    else:
        raise ValueError('unknown smooth')

# --------- 训练 U_psi：使 Lyapunov 不等式成立 ---------
def train_U_lya(
    x0_sampler,
    x1_sampler,
    V_net,                    # 预先 few-shot 学到的 V_theta: (B,2)->(B,1)
    U_base,                   # 已有的 base 流模型 u_t(t,x): concat([x,t]) -> (B,2)
    delta=0.1,
    cfm='cfm',
    sigma=0.0,
    device='cuda:0',
    batch_size=1024,
    steps=20000,
    lr=1e-4,
    width=256,
    where='x1pred',           # 'xt' or 'x1pred'
    reg_U=1e-4,               # L2 正则，防止过激控制
    proj_at_train=False,      # 训练时是否加一次伪投影兜底
):
    """
    训练一个残差控制器 U_psi，使得 <∇V, u+U> + δ V <= 0.
    """
    # 待训控制器 U_psi
    U_net = MLP(dim=2, out_dim=2, w=width, time_varying=True).to(device)
    opt = torch.optim.AdamW(U_net.parameters(), lr=lr)
    FM = get_cfm(cfm, sigma)

    # base 流冻结
    U_base.eval()
    for p in U_base.parameters():
        p.requires_grad_(False)

    V_net.eval()  # V 在本阶段固定；若同时微调 V，可改为 train() 并加入其优化器

    for _ in tqdm.tqdm(range(steps), desc='train_U_lya'):
        # 采样 (x0, x1) 和 t
        x0 = x0_sampler(batch_size=batch_size, device=device)
        x1 = x1_sampler(batch_size=batch_size, device=device)
        t  = torch.rand(x0.shape[0], device=device)

        # CFM 采样位置与条件流
        if cfm == 'ot_cfm':
            t, xt, ut, _, x1 = FM.guided_sample_location_and_conditional_flow(x0, x1, y0=None, y1=x1, t=t) # NOTE: we need correct x1!
        else:
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, t)
        # base 向量场预测
        u_pred = U_base(torch.cat([xt, t[:, None]], dim=-1))            # (B,2)

        # 选取计算 ∇V 的评估点：本地 xt 或外推 x1_pred
        if where == 'xt':
            x_eval = xt
        elif where == 'x1pred':
            x_eval = xt + (1.0 - t)[:, None] * u_pred
        else:
            raise ValueError("where must be 'xt' or 'x1pred'")

        # 计算 V 与 ∇V
        with torch.no_grad():
            Vval = V_net(x_eval).detach()                               # (B,1)
        gradV = grad_V_of(V_net, x_eval).detach()                       # (B,2)

        # 残差控制器输出 U_psi(t, xt)
        U_res = U_net(torch.cat([xt, t[:, None]], dim=-1))              # (B,2)


        # Lyapunov 约束的平滑 hinge 损失
        loss_lya = lya_hinge_loss(gradV, Vval, u_pred, U_res, delta=delta, smooth='relu')

        # 轻度正则，抑制过激控制
        loss_reg = reg_U * (U_res**2).sum(dim=1).mean()

        loss = loss_lya + loss_reg
        opt.zero_grad()
        loss.backward()
        opt.step()

    return U_net

def train_g3(
        x0_sampler,
        x1_sampler,
        # guidance related
        model_v,
        model_z,
        J,
        scale=1.0, 
        # training related
        cfm = 'cfm', 
        width = 64,
        sigma = 0.0,
        device='cuda',
        batch_size=1024,
        lr=1e-4,
        num_steps=20000,
):
    # Model definition
    model = MLP(dim=2, out_dim=2, w=width, time_varying=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    FM = get_cfm(cfm, sigma)
    
    with tqdm.tqdm(range(num_steps)) as pbar:
        for k in pbar:
            optimizer.zero_grad()

            x0 = x0_sampler(batch_size=batch_size, device=device)
            x1 = x1_sampler(batch_size=batch_size, device=device)

            t = torch.rand(x0.shape[0]).to(device)
            if cfm == 'ot_cfm':
                t, xt, ut, _, x1 = FM.guided_sample_location_and_conditional_flow(x0, x1, y0=None, y1=x1, t=t) # NOTE: we need correct x1!
            else:
                t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, t)

            # train guidance g
            z = model_z(torch.cat([xt, t[:, None]], dim=-1))
            v = model_v(torch.cat([xt, t[:, None]], dim=-1))
            g = model(torch.cat([xt, t[:, None]], dim=-1))
            loss = (
                torch.square(
                    g + v - ut  # (B, 2)
                ) * (torch.exp(-scale * J(x1).unsqueeze(-1)) / (z.abs() + 1e-8))
            ).mean()

            loss.backward()
            optimizer.step()

            # Update progress bar with loss
            pbar.set_postfix(loss=loss.item())

    return model

# ========= 3) 端到端示例：先学 V，再学 g_phi =========
def train_s2_pipeline(
    x0_name='uniform',
    x1_name='8gaussian',
    device='cuda',
    seed=0,
    U_base=None,
    # V 训练
    n_support=2048, batch_size_V=256, steps_V=2000, width_V=128, lr_V=1e-3, use_huber=True, grad_reg=0.0,
    # g 训练
    batch_size_g=1024, steps_g=20000, width_g=256, lr_g=1e-4, k_scale=1.0,
    cfm='cfm', sigma=0.0, stop_grad_through_u=True,):
    deterministic(seed)

    x0_dist = get_distribution(x0_name)
    x1_dist = get_distribution(x1_name)

    # --- Step A: 学 V_theta (few-shot) ---
    print(f'learn V: {x0_name} to {x1_name}')
    V_net,Xs,ys = train_V(
        x1_sampler=x1_dist.sample,
        J_true=x1_dist.get_J,      # 这里用 synthetic 的真分数做监督；few-shot 时只用 n_support 个点
        width=width_V, device=device,
        n_support=n_support, batch_size=batch_size_V,
        lr=lr_V, num_steps=steps_V,
        use_huber=use_huber, grad_reg=grad_reg
    )

    # --- Step B: 学 g_phi ---
    print(f'learn G: {x0_name} to {x1_name}')
    G_net = train_U_lya(
        x0_sampler=x0_dist.sample,
        x1_sampler=x1_dist.sample,
        V_net=V_net,
        U_base=U_base,
        cfm=cfm, sigma=sigma, device=device,
        batch_size=batch_size_g,steps=steps_g, lr=lr_g,
    )


    # 保存

    return V_net, G_net, Xs, ys



if __name__ == '__main__':
    cfg = tyro.cli(GuidanceTrainingConfig)
    deterministic(cfg.seed)
    set_cuda_visible_device(cfg)

    model = MLP(dim=2, w=MLP_WIDTH, time_varying=True).to('cuda:0')
    log_subfolder_flow = os.path.join(
        "logs", f'{cfg.x0_dist}-{cfg.x1_dist}/{cfg.cfm}_{cfg.x0_dist}_{cfg.x1_dist}/{cfg.cfm}_{cfg.x0_dist}_{cfg.x1_dist}.pth'
    )

    model.load_state_dict(torch.load(log_subfolder_flow))
  
    V_net, G_net, Xs, ys = train_s2_pipeline(
        x0_name=cfg.x0_dist,
        x1_name=cfg.x1_dist,
        device=cfg.device,
        seed=cfg.seed, U_base=model,
        n_support=cfg.n_support, batch_size_V=256, steps_V=2000, width_V=128, lr_V=1e-3,
        batch_size_g=cfg.batch_size, steps_g=cfg.num_steps, width_g=64, lr_g=cfg.lr,
        cfm=cfg.cfm, sigma=cfg.sigma, k_scale=1.0, stop_grad_through_u=True,
    )

    model_z = MLP(dim=2, out_dim=1, w=cfg.width, time_varying=True, exp_final=True).to(cfg.device)
    try:
        model_z.load_state_dict(torch.load(os.path.join(os.path.join(
        "logs", f"{cfg.x0_dist}-{cfg.x1_dist}", f"{cfg.cfm}_{cfg.x0_dist}_{cfg.x1_dist}"
    ), f"guidance_matching_z_scale_{cfg.scale}_{cfg.x0_dist}_{cfg.x1_dist}.pth")))
    except:
        raise FileNotFoundError("No model_z found!")

    model_v = MLP(dim=2, out_dim=2, w=cfg.width, time_varying=True).to(cfg.device)
    try:
        model_v.load_state_dict(torch.load(os.path.join( os.path.join(
        "logs", f"{cfg.x0_dist}-{cfg.x1_dist}", f"{cfg.cfm}_{cfg.x0_dist}_{cfg.x1_dist}"
    ), f"{cfg.cfm}_{cfg.x0_dist}_{cfg.x1_dist}.pth")))
    except:
        raise FileNotFoundError("No model_v found, training from scratch")  
        
    model_g3 = train_g3(
        get_distribution(cfg.x0_dist).sample, 
        get_distribution(cfg.x1_dist).sample, 
        model_v, 
        model_z, 
        V_net, 
        cfg.scale, 
        cfg.cfm, 
        cfg.width, 
        cfg.sigma, 
        cfg.device, 
        cfg.batch_size, 
        cfg.lr, 
        cfg.num_steps
    )

    # model_path = os.path.join(log_subfolder, f"guidance_matching_g3_scale_{cfg.scale}_{cfg.x0_dist}_{cfg.x1_dist}.pth")
    # torch.save(model_g3.state_dict(), model_path)


    log_subfolder = os.path.join(
        "logs", f"{cfg.x0_dist}-{cfg.x1_dist}", f"{cfg.cfm}_{cfg.x0_dist}_{cfg.x1_dist}", f"n_support={cfg.n_support}"
    )
    os.makedirs(log_subfolder, exist_ok=True)  # 自动创建目录
    print(f'creat {log_subfolder}')

    torch.save(V_net.state_dict(), os.path.join(log_subfolder, f'V_theta_{cfg.x0_dist}_{cfg.x1_dist}.pth'))
    # torch.save(G_net.state_dict(), os.path.join(log_subfolder, f'G_phi_{cfg.x0_dist}_{cfg.x1_dist}.pth'))
    torch.save(model_g3.state_dict(), os.path.join(log_subfolder, f'G_phi_{cfg.x0_dist}_{cfg.x1_dist}.pth'))
    torch.save({'Xs': Xs.cpu(), 'ys': ys.cpu()}, os.path.join(log_subfolder, f'V_theta_{cfg.x0_dist}_{cfg.x1_dist}_support_dataset.pt'))