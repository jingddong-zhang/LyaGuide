from dataclasses import dataclass

@dataclass
class TrainingConfig:
    flow_type: str = 'ot' # ot or cfm
    model_data: str = 'punet256_celeba256' # celeba , imagenet64 , imagenet128
    data_cache_dir: str = 'image'
    batch_size: int = 32 # 64:256*2*2 256:32*2*2 128:64*2
    lr: float = 1e-4
    optimizer: str = 'adam'
    num_epochs: int = 500
    use_ema: bool = False
    lr_scheduler: str = 'None' # 'cosine'
    warmup_epochs: int = 5
    weight_decay: float = 0.0
    seed: int = 0
    device: str = '2,3'
    criterion: str = 'mse'
    load_pretrained_file: str = None
    accumulation_steps: int = 1
    