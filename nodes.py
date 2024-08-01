import base64
from io import BytesIO
from PIL import Image
import numpy as np



class BlackoutMLSD:
    CATEGORY = "utils"

    @classmethod    
    def INPUT_TYPES(cls):
        return { 
            "required":{
                "images": ("IMAGE",),
                "masks": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "func"

    def func(self, images, masks):
        image = images[0]
        image = 255. * image.cpu().numpy()
        image = np.clip(image, 0, 255).astype(np.uint8)
        mask = masks[0].cpu().numpy()

        print(image.shape, mask.shape)
       # Ensure the mask has the same shape as the image (except for the channel dimension)
        if mask.dim() == 2:  # If mask is (H, W)
            mask = mask.unsqueeze(0)  # Add a channel dimension to make it (1, H, W)

        if image.dim() == 3 and mask.dim() == 3:
            mask = mask.expand_as(image)  # Expand the mask to match the image's channels (C, H, W)

        blacked_out_image = image * mask  # Element-wise multiplication

        blacked_out_image = blacked_out_image.unsqueeze(0).float()  # Add batch dimension back
        return (blacked_out_image,)
