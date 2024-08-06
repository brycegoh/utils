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

    def func(self, base_image, image_to_paste, mask):
        base_image = base_image[0]
        base_image = base_image.cpu().numpy()  # Convert to NumPy and scale
        base_image = np.clip(base_image, 0, 255).astype(np.uint8)  # Ensure it's in uint8 format

        image_to_paste = image_to_paste[0]
        image_to_paste = image_to_paste.cpu().numpy()
        image_to_paste = np.clip(image_to_paste, 0, 255).astype(np.uint8)

        mask = mask[0]
        mask = mask.cpu().numpy()
        mask = np.clip(mask, 0, 255).astype(np.uint8)

        if len(mask.shape) == 3 and mask.shape[-1] == 3:  # if mask is in RGB
            mask = mask[:, :, 0]  # Use the red channel or convert to grayscale

        # Normalize the mask to be between 0 and 1, where 255 (white) becomes 1
        mask = mask / 255.0

        # Expand the mask to have the same number of channels as the images
        if len(base_image.shape) == 3 and base_image.shape[-1] > 1:
            mask = np.expand_dims(mask, axis=-1)  # Add a channel dimension
            mask = np.repeat(mask, base_image.shape[-1], axis=-1)  # Repeat across all channels

        # Cut out the region from image_to_paste using the mask
        cut_out = image_to_paste * mask

        # Invert the mask to apply to the base image
        inverse_mask = 1 - mask

        # Blend the cut-out region into the base image
        base_image = base_image * inverse_mask + cut_out

        result_image = torch.from_numpy(base_image).unsqueeze(0).float()

        return result_image
