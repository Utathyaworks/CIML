import torch.nn as nn


class CrossAttentionMultiply(nn.Module):
    """
    Bi-directional cross-attention between z_i (image) and z_m (metadata),
    then element-wise multiply the attended outputs:
        z_i_att = Attn(Q=z_i, K=z_m, V=z_m)
        z_m_att = Attn(Q=z_m, K=z_i, V=z_i)
        z_c = LayerNorm(z_i_att * z_m_att)
    """

    def __init__(self, dim=768, num_heads=8, attn_dropout=0.1):
        super().__init__()
        self.attn_i_to_m = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True
        )
        self.attn_m_to_i = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, z_i, z_m):
        qi = z_i.unsqueeze(1)
        km = z_m.unsqueeze(1)

        z_i_att, _ = self.attn_i_to_m(query=qi, key=km, value=km)
        z_m_att, _ = self.attn_m_to_i(query=km, key=qi, value=qi)

        z_i_att = z_i_att.squeeze(1)
        z_m_att = z_m_att.squeeze(1)

        z_c = z_i_att * z_m_att
        z_c = self.norm(z_c)
        return z_c


class Conv1DLSTMFusion(nn.Module):
    """Conv1D + BiLSTM for temporal-convolutional multimodal fusion."""

    def __init__(self, input_dim=768, hidden_dim=256, num_classes=6):
        super().__init__()
        self.conv1d = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, z_stack):
        x = self.conv1d(z_stack.transpose(1, 2))
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        return self.classifier(x)

