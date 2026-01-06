import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageOps

class FingerprintStandardization:
    def __init__(self, target_size=(512, 512), is_training=True):
        self.target_size = target_size
        self.is_training = is_training

    def place_on_canvas(self, img_pil):
        """
        Centrally places the extracted region on a fixed-size black background.
        - Smaller images are padded with zeros.
        - Larger images are centrally cropped.
        """
        w, h = img_pil.size
        target_w, target_h = self.target_size
        
        # Create fixed-size black background (0 value)
        # The paper specifies "fixed-size black background" 
        canvas = Image.new('L', self.target_size, color=0)
        
        if w > target_w or h > target_h:
            # Case: Larger -> Central Crop
            # "larger ones were centrally cropped" [cite: 231]
            left = (w - target_w) // 2
            top = (h - target_h) // 2
            right = left + target_w
            bottom = top + target_h
            img_crop = img_pil.crop((left, top, right, bottom))
            canvas.paste(img_crop, (0, 0))
        else:
            # Case: Smaller -> Pad (Paste in center)
            # "Images smaller than 512x512 were padded with zeros" [cite: 230-231]
            paste_x = (target_w - w) // 2
            paste_y = (target_h - h) // 2
            canvas.paste(img_pil, (paste_x, paste_y))
            
        return canvas

    def __call__(self, img):
        # --- Step 1: Inversion ---
        # Ensure grayscale
        # if img.mode != 'L':
        #     img = img.convert('L')
            
        # "inverted so that fingerprint ridges appeared bright" 
        img = ImageOps.invert(img)
        
        # --- Step 2: Standardization ---
        # "preprocessing standardizes spatial scale and alignment" [cite: 232]
        processed_img = self.place_on_canvas(img)
        
        # --- Step 3: Data Augmentation ---
        # "In addition, we applied data augmentation during training" [cite: 232-233]
        if self.is_training:
            # "random horizontal and vertical flipping" [cite: 235]
            if torch.rand(1) < 0.5:
                processed_img = F.hflip(processed_img)
            if torch.rand(1) < 0.5:
                processed_img = F.vflip(processed_img)
            
            # "random rotations in the range of -30 to +30" [cite: 235]
            # fill=0 keeps the background black during rotation
            rotation_angle = float(torch.empty(1).uniform_(-30, 30))
            processed_img = F.rotate(processed_img, rotation_angle, fill=[0])

        # Convert to Tensor
        return F.to_tensor(processed_img)

# --- Usage ---
transform1 = {
    # Training: Inverts, Standardizes, and Augments
    'Train': FingerprintStandardization(target_size=(512, 512), is_training=True),

    # Testing: Inverts and Standardizes only
    'Test': FingerprintStandardization(target_size=(512, 512), is_training=False)
}

def get_transforms(transform_type: str):
    if transform_type == 'transform1':
        return transform1