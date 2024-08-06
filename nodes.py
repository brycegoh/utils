import base64
from io import BytesIO
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
import math

def tensor2rgb(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t.unsqueeze(3).repeat(1, 1, 1, 3)
    if size[3] == 1:
        return t.repeat(1, 1, 1, 3)
    elif size[3] == 4:
        return t[:, :, :, :3]
    else:
        return t

def tensor2rgba(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t.unsqueeze(3).repeat(1, 1, 1, 4)
    elif size[3] == 1:
        return t.repeat(1, 1, 1, 4)
    elif size[3] == 3:
        alpha_tensor = torch.ones((size[0], size[1], size[2], 1))
        return torch.cat((t, alpha_tensor), dim=3)
    else:
        return t
    
def tensor2mask(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t
    if size[3] == 1:
        return t[:,:,:,0]
    elif size[3] == 4:
        # Not sure what the right thing to do here is. Going to try to be a little smart and use alpha unless all alpha is 1 in case we'll fallback to RGB behavior
        if torch.min(t[:, :, :, 3]).item() != 1.:
            return t[:,:,:,3]

    return TF.rgb_to_grayscale(tensor2rgb(t).permute(0,3,1,2), num_output_channels=1)[:,0,:,:]



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
        image = image.cpu().numpy()  # Convert to NumPy and scale
        image = np.clip(image, 0, 255).astype(np.uint8)  # Ensure it's in uint8 format
        mask = masks[0].cpu().numpy()  # Convert mask to NumPy

        # resize mask to match image
        mask = Image.fromarray(mask)
        mask = mask.resize((image.shape[1], image.shape[0]), Image.LANCZOS)
        mask = np.array(mask)

        print(image.shape, mask.shape)  # This should print (896, 1152, 3) (896, 1152)
        # print unique values in mask
        print(np.unique(mask))
        # Ensure the mask has the same shape as the image (except for the channel dimension)
        if len(mask.shape) == 2:  # If mask is (H, W)
            mask = np.expand_dims(mask, axis=-1)  # Add a channel dimension to make it (H, W, 1)

        # Expand the mask to match the image's channels
        mask = np.repeat(mask, 3, axis=-1)  # Expand mask to have the same number of channels as the image (H, W, 3)

        # Set the masked areas to black
        blacked_out_image = image.copy()  # Create a copy of the image to modify
        blacked_out_image[mask == 0] = 0  # Set pixels where mask is 0 to black

        # Convert back to torch tensor if needed
        blacked_out_image = torch.from_numpy(blacked_out_image).unsqueeze(0).float()  # Add batch dimension back and convert to float
        return (blacked_out_image,)


class PasteMask:
    CATEGORY = "utils"

    @classmethod    
    def INPUT_TYPES(cls):
        return { 
            "required":{
                "image_base": ("IMAGE",),
                "image_to_paste": ("IMAGE",),
                "mask": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "func"

    def func(self, image_base, image_to_paste, mask):
        image_base = tensor2rgba(image_base)
        image_to_paste = tensor2rgba(image_to_paste)
        mask = tensor2mask(mask)

        # Normalize the mask to [0, 1]
        mask = mask / 255.0

        # Convert the mask to 4 channels to match RGBA format
        mask_repeated = mask.unsqueeze(-1).repeat(1, 1, 1, 4)  # Repeat the mask across the RGBA channels

        # Create an inverted mask
        inverted_mask = 1.0 - mask_repeated

        # Cut out sections from image_to_paste where the mask is non-black
        cut_out = image_to_paste * mask_repeated

        # Paste the cut-out over image_base using the mask
        result = image_base * inverted_mask + cut_out

        # Ensure the result has a batch size of 1
        result = result.unsqueeze(0)  # Add batch dimension

        return (result,)


class PrintImageSize:
    CATEGORY = "utils"

    @classmethod    
    def INPUT_TYPES(cls):
        return { 
            "required":{
                "image": ("IMAGE",),
                "prefix": ("STRING", {"default": "image"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "func"

    def func(self, image, prefix):
        print(f"{prefix}: {image.shape}")
        return (image,)
