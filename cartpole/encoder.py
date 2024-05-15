import torch.nn as nn


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


class MLPEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, *args, **kwargs):
        super().__init__()
        assert len(obs_shape) == 1
        obs_shape = obs_shape[0]
        
        layers = nn.ModuleList()
        layers.append(nn.Linear(obs_shape, feature_dim))

        for _ in range(num_layers - 1):
            layers.append(nn.LeakyReLU())
            layers.append(nn.Linear(feature_dim, feature_dim))
        
        self.model = nn.Sequential(*layers)
        self.feature_dim = feature_dim
        self.max_norm = kwargs.get('max_norm')

    def forward(self, obs, detach=False, normalize=True):
        x = self.model(obs)
        if self.max_norm and normalize:
            x = self.normalize(x)
        if detach:
            x = x.detach()
        return x
    
    def normalize(self, x):
        norms = x.norm(dim=-1)
        norm_to_max = (norms / self.max_norm).clamp(min=1).unsqueeze(-1)
        x = x / norm_to_max
        return x
    
    def copy_conv_weights_from(self, source):
        source_layers = [m for m in source.modules() if isinstance(m, nn.Linear)]
        self_layers = [m for m in self.modules() if isinstance(m, nn.Linear)]
        assert len(self_layers) == len(source_layers)
        for self_layer, source_layer in zip(self_layers, source_layers):
            tie_weights(src=source_layer, trg=self_layer)
