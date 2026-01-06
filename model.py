import torch
import torch.nn as nn

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        # The Spatial Attention Module uses a 7x7 kernel with 98 parameters.
        # 98 params = 2 (input channels: max + avg) * 1 (output channel) * 49 (kernel)
        # Bias is False to match the exact parameter count (98).
        # Padding is 3 to maintain the spatial dimension (28x28) given a 7x7 kernel.
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Generate average and max pool features along the channel dimension
        # Input shape: [Batch, 32, 28, 28] -> Output pooling: [Batch, 1, 28, 28]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate to form a 2-channel feature map
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        # Convolve to generate spatial attention map
        attn_map = self.conv(x_cat)
        attn_map = self.sigmoid(attn_map)
        
        # Element-wise multiplication to refine features
        return x * attn_map

class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        
        # --- Feature Extraction Module ---
        
        # Layer 1: Conv2d-1 -> MaxPool2d-1 -> ReLU-1
        # Input: 1 x 512 x 512
        # Conv2d-1: 1 -> 128, 7x7, stride 2. Params: (128*1*7*7) + 128 bias = 6400 [cite: 200]
        # Output calculation: floor((512 - 7)/2) + 1 = 253
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=7, stride=2, padding=0), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        
        # Layer 2: Conv2d-2 -> MaxPool2d-2 -> ReLU-2
        # Input: 128 x 126 x 126
        # Conv2d-2: 128 -> 64, 3x3, stride 1. Params: (64*128*3*3) + 64 bias = 73,792 [cite: 200]
        # Output calculation: (126 - 3)/1 + 1 = 124
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        
        # Layer 3: Conv2d-3 -> MaxPool2d-3 -> ReLU-3
        # Input: 64 x 62 x 62
        # Conv2d-3: 64 -> 64, 3x3, stride 1. Params: (64*64*3*3) + 64 bias = 36,928 [cite: 200]
        # Output calculation: (62 - 3)/1 + 1 = 60
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        
        # Layer 4: Conv2d-4
        # Input: 64 x 30 x 30
        # Conv2d-4: 64 -> 32, 3x3, stride 1. Params: (32*64*3*3) + 32 bias = 18,464 [cite: 200]
        # Output calculation: (30 - 3)/1 + 1 = 28
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0)
        
        # --- Spatial Attention Module ---
        # Input: 32 x 28 x 28
        # Params: 98 [cite: 200]
        self.sam = SpatialAttentionModule()
        
        # --- Final Convolutional Module ---
        
        # Layer 5: Conv2d-5 -> ReLU-4
        # Input: 32 x 28 x 28
        # Conv2d-5: 32 -> 16, 3x3, stride 1. Params: (16*32*3*3) + 16 bias = 4624 [cite: 200]
        # Output calculation: (28 - 3)/1 + 1 = 26
        self.layer5 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        
        # Layer 6: Conv2d-6 -> TanH
        # Input: 16 x 26 x 26
        # Conv2d-6: 16 -> 1, 3x3, stride 1. Params: (1*16*3*3) + 1 bias = 145 [cite: 200]
        # Output calculation: (26 - 3)/1 + 1 = 24
        self.layer6 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        # Feature Extraction
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.conv4(x)
        
        # Attention Mechanism [cite: 201]
        x = self.sam(x)
        
        # Final Classification Block
        x = self.layer5(x)
        x = self.layer6(x)
        
        # Global Mean Pooling to obtain scalar output [cite: 200, 216]
        # Output shape is 1x24x24, mean pooling reduces it to a single scalar per batch item
        x = torch.mean(x, dim=[1, 2, 3])
        
        return x

def get_model():
    return Model1()