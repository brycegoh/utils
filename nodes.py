import base64
from io import BytesIO
from PIL import Image
import numpy as np
import torch



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
                "base_image": ("IMAGE",),
                "image_to_paste": ("IMAGE",),
                "mask": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "func"

    def func(self, image_base, image_to_paste, mask):
        # Assume all images and mask are of the same size and already in the required format
        image_base = image_base[0].cpu().numpy()  # Convert base image to NumPy
        image_to_paste = image_to_paste[0].cpu().numpy()  # Convert paste image to NumPy
        mask = mask[0].cpu().numpy()  # Convert mask to NumPy

        # Ensure the mask is normalized to [0, 1]
        mask = mask / 255.0

        # Convert the images and mask to torch tensors (if needed)
        image_base = torch.tensor(image_base).float()
        image_to_paste = torch.tensor(image_to_paste).float()
        mask = torch.tensor(mask).float()

        # Get the size of the images and mask
        B, H, W, C = image_base.shape

        # Create an empty result image
        result = image_base.clone()

        # Apply the mask to image_to_paste and blend with the base image
        for i in range(B):
            mask_i = mask[i].unsqueeze(-1).repeat(1, 1, C)  # Repeat mask for each channel
            inverted_mask = 1.0 - mask_i

            # Blend the cut-out portion of image_to_paste into the base image
            result[i] = image_base[i] * inverted_mask + image_to_paste[i] * mask_i

        # Convert result back to a format with batch size of 1
        result = result.unsqueeze(0)  # Add batch dimension

        return (result,)

        # Blend the cut-out region into the base image
        base_image = base_image * inverse_mask + cut_out

        result_image = torch.from_numpy(base_image).unsqueeze(0).float()

        return (result_image,)
