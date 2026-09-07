import torch.nn as nn

class PointFeatureEmbedDeep(nn.Module):
    def __init__(self, in_dim=3, out_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        x: (M, 3)
        """
        x = x.unsqueeze(0).transpose(1, 2)  # (1,3,M)
        x = self.proj(x)                    # (1,256,M)
        return x

