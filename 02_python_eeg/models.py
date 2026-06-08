import torch
import torch.nn as nn
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class EEGSNet(nn.Module):
    def __init__(self, input_shape=(1, 3, 76, 60), num_classes=5, lstm_hidden=128):
        # Input shape: (Batch, Channels, Freq, Time) -> (B, 3, 76, 60)
        # Assuming we transpose the Matlab (76, 60, 3) to (3, 76, 60)
        super(EEGSNet, self).__init__()
        
        self.dropout_prob = 0.5
        
        # --- Block 1 ---
        self.b1_conv1 = nn.Conv2d(3, 16, kernel_size=3, padding='same')
        self.b1_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding='same')
        self.b1_conv3 = nn.Conv2d(32, 32, kernel_size=3, padding='same')
        self.b1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.b1_bn = nn.BatchNorm2d(32)
        self.b1_drop = nn.Dropout(self.dropout_prob)
        
        # --- Block 2 ---
        self.b2_conv1 = nn.Conv2d(3, 16, kernel_size=1, stride=1) # Note: Connected to INPUT in Matlab
        self.b2_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding='same')
        self.b2_conv3 = nn.Conv2d(32, 32, kernel_size=3, padding='same')
        self.b2_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1) # Matlab: Padding 'same' for avg pool means keeping size approx same? 3x3 stride 1 needs pad 1
        self.b2_bn = nn.BatchNorm2d(32)
        self.b2_drop = nn.Dropout(self.dropout_prob)
        
        # Li 2022: Coordinate Attention after Block 2
        self.ca2 = CoordAtt(32, 32)

        # Res Connection 1 Logic
        # Matlab: b2_res_downsample (Conv 1x1, 32 filters, stride 2) after b2_dropout
        self.b2_res_downsample = nn.Conv2d(32, 32, kernel_size=1, stride=2)
        
        # --- Block 3 ---
        # Input: Add1 (b1_dropout + b2_res_downsample)
        self.b3_conv1 = nn.Conv2d(32, 20, kernel_size=1, stride=1)
        self.b3_conv2 = nn.Conv2d(20, 64, kernel_size=3, padding='same')
        self.b3_conv3 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.b3_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.b3_bn = nn.BatchNorm2d(64)
        self.b3_drop = nn.Dropout(self.dropout_prob)

        # --- Block 4 ---
        # Input: Add1
        self.b4_conv1 = nn.Conv2d(32, 20, kernel_size=1, stride=1)
        self.b4_conv2 = nn.Conv2d(20, 32, kernel_size=3, padding='same')
        self.b4_conv3 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
        self.b4_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_bn = nn.BatchNorm2d(64)
        self.b4_drop = nn.Dropout(self.dropout_prob)
        
        # Li 2022: Coordinate Attention after Block 4
        self.ca4 = CoordAtt(64, 64)

        # Res Connection 2 Logic
        # Matlab: b4_res_downsample (Conv 1x1, 64 filters, stride 2)
        self.b4_res_downsample = nn.Conv2d(64, 64, kernel_size=1, stride=2)

        # --- Block 5 ---
        # Input: Add2 (b3_dropout + b4_res_downsample)
        self.b5_conv1 = nn.Conv2d(64, 20, kernel_size=1, stride=1)
        self.b5_conv2 = nn.Conv2d(20, 64, kernel_size=3, padding='same')
        self.b5_conv3 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.b5_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.b5_bn = nn.BatchNorm2d(64)
        self.b5_drop = nn.Dropout(self.dropout_prob)
        
        # Global Average Pooling
        # In PyTorch: AdaptiveAvgPool2d(1) -> Flatten
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        
        # --- LSTM ---
        # Input size to LSTM: 64 (from b5/gap)
        self.lstm = nn.LSTM(input_size=64, hidden_size=lstm_hidden, num_layers=2, batch_first=True, bidirectional=True)
        # Note: Matlab uses bilstm1 (sequence output) -> bilstm2 (last output)
        # PyTorch LSTM returns output (seq), (h_n, c_n).
        # We need the LAST time step. But wait, is the input a sequence?
        # In the Matlab model, 'imageInputLayer' is used. It's a single image.
        # Why is LSTM used? 
        # Ah, in Matlab: 
        # lgraph = connectLayers(lgraph, 'flatten_gap', 'bilstm1');
        # `flatten_gap` output is 64 features.
        # If input is a single image, "sequence length" is 1? Or does it verify temporal features?
        # Actually, `imageInputLayer` treats it as an image.
        # However, `bilstmLayer` expects sequence. If connected after flatten, Matlab treats the vector as a sequence of length 1 or converts feature dim to sequence?
        # Wait, `flatten_gap` produces a vector. 
        # If we pass (N, 64) to LSTM, we need to reshape to (N, SequenceLength, Features).
        # Typically in this hybrid architecture, if we don't have multiple time steps of EEG *spectrograms*, maybe we are treating the feature vector as a sequence?
        # Let's check Matlab `defineEEGSNet.m`:
        # input: [76 60 3] -> Image.
        # GAP -> vector.
        # LSTM input: vector?
        # This implies standard ResNet-like CNN feature extraction.
        # The use of LSTM on a single image's features is odd unless we are processing a sequence of Epochs.
        # BUT `trainNetwork` uses `arrayDatastore`. 
        # Ah, maybe the input is (Height, Width, Channel, Batch).
        # If it's single-epoch classification, the LSTM might be used just for its gating mechanism on the feature vector (SeqLen=1).
        # Let's assume Sequence Length = 1.
        
        self.fc = nn.Linear(lstm_hidden * 2, num_classes) # Bidirectional = 2 * hidden
        
        # Auxiliary Classifier (Li et al. 2022)
        # Attaches to Feature Extractor output (before LSTM).
        # Input: 64 features.
        self.aux_fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x shape: (Batch, Seq, Channels, Freq, Time) or (Batch, Channels, Freq, Time)
        # We need to handle both for compatibility, or enforce Sequence.
        
        if x.dim() == 5:
            # Sequence Input: (B, S, C, F, T)
            B, S, C, n_freq, n_time = x.size()
            # Merge dimensions for CNN: (B*S, C, F, T)
            input_data = x.view(B * S, C, n_freq, n_time)
        else:
            # Single Epoch Input: (B, C, F, T) - Backwards compatibility
            # Treat as Seq=1
            input_data = x
            B = x.size(0)
            S = 1
        
        # Block 1
        x1 = F.gelu(self.b1_conv1(input_data))
        x1 = F.gelu(self.b1_conv2(x1))
        x1 = F.gelu(self.b1_conv3(x1))
        x1 = self.b1_drop(self.b1_bn(self.b1_pool(x1)))
        
        # Block 2 (Connected to Input)
        x2 = F.gelu(self.b2_conv1(input_data))
        x2 = F.gelu(self.b2_conv2(x2))
        x2 = F.gelu(self.b2_conv3(x2))
        x2 = self.b2_drop(self.b2_bn(self.b2_pool(x2)))
        
        # Apply CA2
        x2 = self.ca2(x2)
        
        # Downsample B2
        x2_down = self.b2_res_downsample(x2)
        
        # Add1
        add1 = x1 + x2_down
        
        # Block 3
        x3 = F.gelu(self.b3_conv1(add1))
        x3 = F.gelu(self.b3_conv2(x3))
        x3 = F.gelu(self.b3_conv3(x3))
        x3 = self.b3_drop(self.b3_bn(self.b3_pool(x3)))
        
        # Block 4 (Connected to Add1)
        x4 = F.gelu(self.b4_conv1(add1))
        x4 = F.gelu(self.b4_conv2(x4))
        x4 = F.gelu(self.b4_conv3(x4))
        x4 = self.b4_drop(self.b4_bn(self.b4_pool(x4)))
        
        # Apply CA4
        x4 = self.ca4(x4)
        
        # Downsample B4
        x4_down = self.b4_res_downsample(x4)
        
        # Add2
        add2 = x3 + x4_down
        
        # Block 5
        x5 = F.gelu(self.b5_conv1(add2))
        x5 = F.gelu(self.b5_conv2(x5))
        x5 = F.gelu(self.b5_conv3(x5))
        x5 = self.b5_drop(self.b5_bn(self.b5_pool(x5)))
        
        # GAP
        gap = self.gap(x5) # (B*S, 64, 1, 1)
        feat = self.flatten(gap) # (B*S, 64)
        
        # Auxiliary Output (Computed on individual epochs in the sequence, or just the last one?)
        # Li 2022 says "auxiliary classifier after feature extraction".
        # If we have a sequence, we probably train Aux on ALL epochs in the sequence to maximize supervison?
        # Or just the target one?
        # Let's compute it for ALL items in batch*seq for now.
        aux_out_all = self.aux_fc(feat) # (B*S, num_classes)
        
        # For loss calculation, we might want to aggregate or just return (B*S, 5) and flatten labels too.
        # But wait, main output `out` is usually 1-per-sequence.
        # Let's reshape aux_out to (B, S, 5).
        # We can figure out how to calculate loss in train.py (maybe average over sequence?).
        aux_out = aux_out_all.view(B, S, -1)
        
        # LSTM
        # Input (B, Seq, Feat)
        lstm_in = feat.view(B, S, -1)
        lstm_out, _ = self.lstm(lstm_in) # (B, Seq, 256)
        
        # Last step
        last_step = lstm_out[:, -1, :] # (B, 256)
        
        # Classifier
        out = self.fc(last_step) # (B, 5)
        
        return out, aux_out
