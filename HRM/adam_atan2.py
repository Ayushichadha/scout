import torch
class AdamATan2(torch.optim.Adam):
    """CPU stub fallback: behaves like torch.optim.Adam."""
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0.0, **kw):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
