import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# IMPROVED LINKNET BLOCK
# ============================================================

class LinkNetBlock(nn.Module):
    """
    1D LinkNet-style feature extraction block for
    tabular software metrics.

    The original LinkNet concept uses encoder-decoder
    connections. Here the same idea is adapted to a
    one-dimensional feature sequence.
    """

    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm1d(
            out_channels
        )

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm1d(
            out_channels
        )

        # Residual / link connection
        if in_channels != out_channels:

            self.shortcut = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1
            )

        else:

            self.shortcut = nn.Identity()

    def forward(self, x):

        identity = self.shortcut(x)

        x = self.conv1(x)

        x = self.bn1(x)

        x = F.relu(x)

        x = self.conv2(x)

        x = self.bn2(x)

        x = x + identity

        x = F.relu(x)

        return x


# ============================================================
# IMPROVED LINKNET
# ============================================================

class ImprovedLinkNet(nn.Module):
    """
    LinkNet-style neural network adapted for software
    defect prediction from numerical software metrics.
    """

    def __init__(
        self,
        num_features,
        hidden_channels=64,
        num_classes=2
    ):
        super().__init__()

        self.num_features = num_features

        # Initial feature projection
        self.input_layer = nn.Conv1d(
            1,
            hidden_channels,
            kernel_size=3,
            padding=1
        )

        self.input_bn = nn.BatchNorm1d(
            hidden_channels
        )

        # Encoder blocks
        self.encoder1 = LinkNetBlock(
            hidden_channels,
            hidden_channels
        )

        self.encoder2 = LinkNetBlock(
            hidden_channels,
            hidden_channels * 2
        )

        # Decoder blocks
        self.decoder1 = LinkNetBlock(
            hidden_channels * 2,
            hidden_channels
        )

        self.decoder2 = LinkNetBlock(
            hidden_channels,
            hidden_channels
        )

        # Global feature representation
        self.pool = nn.AdaptiveAvgPool1d(
            1
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                hidden_channels,
                hidden_channels
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                hidden_channels,
                num_classes
            )
        )

    def forward(self, x):

        # x:
        # [batch, features]

        x = x.unsqueeze(1)

        # [batch, 1, features]

        x = self.input_layer(x)

        x = self.input_bn(x)

        x = F.relu(x)

        # Encoder
        x = self.encoder1(x)

        x = self.encoder2(x)

        # Decoder
        x = self.decoder1(x)

        x = self.decoder2(x)

        # Global pooling
        x = self.pool(x)

        # [batch, channels, 1]
        x = x.squeeze(-1)

        # Classification
        logits = self.classifier(x)

        return logits


# ============================================================
# BI-LSTM
# ============================================================

class BiLSTMModel(nn.Module):
    """
    Bidirectional LSTM for sequential representation
    of software metric features.
    """

    def __init__(
        self,
        num_features,
        hidden_size=64,
        num_layers=2,
        num_classes=2
    ):
        super().__init__()

        self.num_features = num_features

        self.lstm = nn.LSTM(

            input_size=1,

            hidden_size=hidden_size,

            num_layers=num_layers,

            batch_first=True,

            bidirectional=True,

            dropout=0.3
            if num_layers > 1
            else 0.0
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                hidden_size * 2,
                hidden_size
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                hidden_size,
                num_classes
            )
        )

    def forward(self, x):

        # x:
        # [batch, features]

        # Convert every feature into a
        # one-dimensional sequence element.

        x = x.unsqueeze(-1)

        # [batch, features, 1]

        output, _ = self.lstm(x)

        # Use the final sequence representation

        x = output[:, -1, :]

        logits = self.classifier(x)

        return logits


# ============================================================
# HYBRID MODEL
# ============================================================

class HybridDefectModel(nn.Module):
    """
    Hybrid model:

        Improved LinkNet
              +
            Bi-LSTM
              ↓
        Score-Level Fusion
              ↓
            Softmax
    """

    def __init__(
        self,
        num_features,
        linknet_weight=0.5,
        bilstm_weight=0.5
    ):
        super().__init__()

        self.num_features = num_features

        self.linknet_weight = linknet_weight

        self.bilstm_weight = bilstm_weight

        self.linknet = ImprovedLinkNet(
            num_features=num_features
        )

        self.bilstm = BiLSTMModel(
            num_features=num_features
        )

    def forward(self, x):

        # ----------------------------------------------------
        # LinkNet
        # ----------------------------------------------------

        linknet_logits = self.linknet(x)

        # ----------------------------------------------------
        # Bi-LSTM
        # ----------------------------------------------------

        bilstm_logits = self.bilstm(x)

        # ----------------------------------------------------
        # Convert each branch into probabilities
        # ----------------------------------------------------

        linknet_probability = F.softmax(
            linknet_logits,
            dim=1
        )

        bilstm_probability = F.softmax(
            bilstm_logits,
            dim=1
        )

        # ----------------------------------------------------
        # Score-Level Fusion
        # ----------------------------------------------------

        fused_probability = (

            self.linknet_weight
            * linknet_probability

            +

            self.bilstm_weight
            * bilstm_probability
        )

        # Normalize the fused scores
        fused_probability = (
            fused_probability
            /
            fused_probability.sum(
                dim=1,
                keepdim=True
            )
        )

        # Convert fused probabilities to logits
        # for compatibility with loss functions.

        fused_logits = torch.log(
            fused_probability.clamp(
                min=1e-8
            )
        )

        return {

            "linknet_logits":
                linknet_logits,

            "bilstm_logits":
                bilstm_logits,

            "linknet_probabilities":
                linknet_probability,

            "bilstm_probabilities":
                bilstm_probability,

            "fused_logits":
                fused_logits,

            "fused_probability":
                fused_probability
        }

    def predict(self, x):

        self.eval()

        with torch.no_grad():

            outputs = self.forward(x)

            probabilities = (
                outputs[
                    "fused_probability"
                ]
            )

            prediction = torch.argmax(
                probabilities,
                dim=1
            )

            confidence = torch.max(
                probabilities,
                dim=1
            ).values

        return {

            **outputs,

            "probabilities":
                probabilities,

            "prediction":
                prediction,

            "confidence":
                confidence
        }