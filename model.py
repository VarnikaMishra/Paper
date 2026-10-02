import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# DILATED LINKNET BLOCK (EQUATIONS 1 & 2)
# ============================================================

class DilatedLinkNetBlock(nn.Module):
    """
    Improved LinkNet Block with Dilated Convolutions & Residual Connections.
    
    Paper Equation (1):
        ks_eff = dr * (ks - 1) + 1
        where dr is the dilation rate and ks is the original kernel size.
        
    Paper Equation (2):
        rf_n = x_{n-1} * rf_{n-1} + ks_{n-1} - x_{n-1}
        receptive field expansion across layers without increasing parameter count.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation_rate=2):
        super().__init__()

        self.dilation_rate = dilation_rate
        # Calculate padding to keep output length consistent: padding = dilation * (kernel_size - 1) // 2
        padding = dilation_rate * (kernel_size - 1) // 2

        # First dilated convolution
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation_rate,
            padding=padding
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        # Second standard convolution
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Residual shortcut / link connection
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Residual Addition (LinkNet direct connection)
        out = out + identity
        out = F.relu(out)

        return out


# ============================================================
# IMPROVED LINKNET (ENCODER-DECODER WITH SKIP CONNECTIONS)
# ============================================================

class ImprovedLinkNet(nn.Module):
    """
    Improved LinkNet for Software Defect Feature Extraction and Classification.
    
    Features:
    - Primary 1D convolution with initial receptive field expansion
    - Multi-stage Encoder with Dilated Convolutions
    - Multi-stage Decoder with direct skip connections bypassing encoder layers to decoder outputs
    - Global feature pooling and classification head
    """

    def __init__(self, input_dim, hidden_channels=64, num_classes=2):
        super().__init__()
        self.input_dim = input_dim

        # Initial stem convolution
        self.stem_conv = nn.Conv1d(1, hidden_channels, kernel_size=7, stride=1, padding=3)
        self.stem_bn = nn.BatchNorm1d(hidden_channels)

        # --- Encoder Stages ---
        # Encoder 1: Dilation rate = 1 (local patterns)
        self.enc1 = DilatedLinkNetBlock(hidden_channels, hidden_channels, dilation_rate=1)
        
        # Encoder 2: Dilation rate = 2 (expanded receptive field)
        self.enc2 = DilatedLinkNetBlock(hidden_channels, hidden_channels * 2, dilation_rate=2)
        
        # Encoder 3: Dilation rate = 4 (high-level structural dependencies)
        self.enc3 = DilatedLinkNetBlock(hidden_channels * 2, hidden_channels * 4, dilation_rate=4)

        # --- Decoder Stages with Skip Connections ---
        # Decoder 3 (matches Encoder 3)
        self.dec3 = DilatedLinkNetBlock(hidden_channels * 4, hidden_channels * 2, dilation_rate=2)
        
        # Decoder 2 (matches Encoder 2) + Skip connection from Enc2
        self.dec2 = DilatedLinkNetBlock(hidden_channels * 2, hidden_channels, dilation_rate=1)
        
        # Decoder 1 (matches Encoder 1) + Skip connection from Enc1
        self.dec1 = DilatedLinkNetBlock(hidden_channels, hidden_channels, dilation_rate=1)

        # Global average pooling & Classifier Head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels // 2, num_classes)
        )

    def forward(self, x):
        # x: [batch, features] -> [batch, 1, features]
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Stem
        x0 = F.relu(self.stem_bn(self.stem_conv(x)))

        # Encoder forward pass
        e1 = self.enc1(x0)            # Stage 1
        e2 = self.enc2(e1)            # Stage 2
        e3 = self.enc3(e2)            # Stage 3 (Bottleneck)

        # Decoder forward pass with Skip Connections
        d3 = self.dec3(e3)            # Channels: 4C -> 2C
        d3 = d3 + e2                  # Skip connection from Encoder 2

        d2 = self.dec2(d3)            # Channels: 2C -> C
        d2 = d2 + e1                  # Skip connection from Encoder 1

        d1 = self.dec1(d2)            # Channels: C -> C
        d1 = d1 + x0                  # Skip connection from Stem

        # Global Feature Pooling
        feat = self.global_pool(d1).squeeze(-1)

        # Classification Logits
        logits = self.classifier(feat)
        return logits


# ============================================================
# BI-LSTM (EQUATIONS 3 TO 10)
# ============================================================

class BiLSTMModel(nn.Module):
    """
    Bidirectional Long Short-Term Memory Network for Software Defect Prediction.
    
    Implements Equations (3)-(10):
    - Forward & Backward LSTM cells capturing bidirectional context
    - Forget Gate (Eq 3), Input Gate (Eq 4-5), Cell State (Eq 6), Output Gate (Eq 7-8)
    - Concatenated hidden state h_t = [h_forward; h_backward] (Eq 9)
    - Softmax Output Classification: y = softmax(W_y * h_t + b_y) (Eq 10)
    """

    def __init__(self, input_dim, hidden_size=64, num_layers=2, num_classes=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size

        # Bidirectional LSTM Layer
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0.0
        )

        # Output Linear Projection (Eq 10)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # x: [batch, features] -> [batch, seq_len=features, input_size=1]
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        lstm_out, (h_n, _) = self.lstm(x)

        # Concatenate forward and backward final hidden states (Equation 9)
        # h_n shape: [num_layers * 2, batch, hidden_size]
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        h_combined = torch.cat((h_forward, h_backward), dim=1)  # [batch, hidden_size * 2]

        # Final classification logits (Equation 10)
        logits = self.classifier(h_combined)
        return logits


# ============================================================
# HYBRID SCORE-LEVEL FUSION MODEL (SECTION 4.6, EQ 11)
# ============================================================

class HybridDefectModel(nn.Module):
    """
    Hybrid Deep Learning Architecture for Software Defect Classification:
        Code2Vec / Features 
             ├──> Improved LinkNet (Spatial Structural Features with Dilated Convolutions)
             └──> Bi-LSTM (Temporal / Sequential Code Features)
                       │
                       ▼
             Score-Level Fusion (Weighted Averaging Eq 11 & Learned Meta-Model)
                       │
                       ▼
             Softmax Binary Defect Classification
    """

    def __init__(
        self,
        num_features,
        linknet_weight=0.5,
        bilstm_weight=0.5,
        use_learned_fusion=False
    ):
        super().__init__()
        self.num_features = num_features
        self.linknet_weight = linknet_weight
        self.bilstm_weight = bilstm_weight
        self.use_learned_fusion = use_learned_fusion

        # Model branches
        self.linknet = ImprovedLinkNet(input_dim=num_features, hidden_channels=64, num_classes=2)
        self.bilstm = BiLSTMModel(input_dim=num_features, hidden_size=64, num_layers=2, num_classes=2)

        # Optional Learned Meta-Model Fusion (Section 4.6.2)
        if use_learned_fusion:
            self.fusion_meta_model = nn.Sequential(
                nn.Linear(4, 8),
                nn.ReLU(),
                nn.Linear(8, 2)
            )

    def forward(self, x):
        # 1. Forward pass through Improved LinkNet
        linknet_logits = self.linknet(x)
        linknet_probs = F.softmax(linknet_logits, dim=1)

        # 2. Forward pass through Bi-LSTM
        bilstm_logits = self.bilstm(x)
        bilstm_probs = F.softmax(bilstm_logits, dim=1)

        # 3. Score-Level Fusion
        if self.use_learned_fusion:
            # Learned Fusion Meta-Model (Section 4.6.2)
            concat_scores = torch.cat([linknet_probs, bilstm_probs], dim=1)
            fused_logits = self.fusion_meta_model(concat_scores)
            fused_probability = F.softmax(fused_logits, dim=1)
        else:
            # Weighted Averaging Fusion (Equation 11): S_final = w1 * S_linknet + w2 * S_Bi-LSTM
            fused_probability = (
                self.linknet_weight * linknet_probs +
                self.bilstm_weight * bilstm_probs
            )
            # Normalize to guarantee probability distribution summing to 1
            fused_probability = fused_probability / (fused_probability.sum(dim=1, keepdim=True) + 1e-8)
            fused_logits = torch.log(fused_probability.clamp(min=1e-8))

        return {
            "linknet_logits": linknet_logits,
            "bilstm_logits": bilstm_logits,
            "linknet_probabilities": linknet_probs,
            "bilstm_probabilities": bilstm_probs,
            "fused_logits": fused_logits,
            "fused_probability": fused_probability
        }

    def predict(self, x):
        """Evaluation and inference wrapper."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            probabilities = outputs["fused_probability"]
            prediction = torch.argmax(probabilities, dim=1)
            confidence = torch.max(probabilities, dim=1).values

        return {
            **outputs,
            "probabilities": probabilities,
            "prediction": prediction,
            "confidence": confidence
        }