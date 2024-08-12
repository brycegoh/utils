import base64
from io import BytesIO
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
import math
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
import os

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

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "cut_out")
    FUNCTION = "func"

    def func(self, image_base, image_to_paste, mask):
        combined_images = []
        cut_out_images = []

        base_image = image_base[0]
        if isinstance(base_image, torch.Tensor):
            base_image = 255. * base_image.cpu().numpy()  # Convert to NumPy and scale
        else:
            base_image = 255. * base_image 
        base_image = np.clip(base_image, 0, 255).astype(np.uint8)  # Ensure it's in uint8 format

        mask = mask[0]
        if isinstance(mask, torch.Tensor):
            mask = 255. * mask.cpu().numpy()
        else:
            mask = 255. * mask  # If it's already a NumPy array
        mask = np.clip(mask, 0, 255).astype(np.uint8)

        # Ensure mask is correctly shaped
        boolean_mask = mask > 0  # Create the boolean mask here once
        boolean_mask = boolean_mask.astype(np.uint8)  # Convert boolean mask to uint8 for multiplication

        for i in range(len(image_to_paste)):
            img = image_to_paste[i]
            if isinstance(img, torch.Tensor):
                img = 255. * img.cpu().numpy()
            else:
                img = 255. * img 
            img = np.clip(img, 0, 255).astype(np.uint8)
            
            # Ensure the boolean mask matches the image shape
            assert boolean_mask.shape == img.shape == base_image.shape, \
                f"Boolean mask shape {boolean_mask.shape} does not match image_to_paste shape {img.shape} or base_image shape {base_image.shape}"
            
            # Cut out the part of image_to_paste where mask is non-zero
            cut_out_image = img * boolean_mask
            
            # Paste the cut out onto the base image
            combined_image = base_image * (1 - boolean_mask) + cut_out_image
            
            # Convert the final image back to the format needed for further processing
            combined_image = torch.from_numpy(combined_image / 255.0).unsqueeze(0)  # Add batch dimension
            cut_out_image = torch.from_numpy(cut_out_image / 255.0).unsqueeze(0)  # Add batch dimension
            
            # Store the results for each image
            combined_images.append(combined_image)
            cut_out_images.append(cut_out_image)

        # Stack the results to create tensors of shape (batch_size, ...)
        combined_images = torch.cat(combined_images, dim=0)
        cut_out_images = torch.cat(cut_out_images, dim=0)

        return (combined_images, cut_out_images)


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

class OneformerDinatSegmentation:
    CATEGORY = "utils"

    @classmethod    
    def INPUT_TYPES(cls):
        return { 
            "required":{
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "func"

    def func(self, image):
        palette = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
                [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
                [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
                [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
                [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
                [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
                [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
                [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
                [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
                [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
                [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
                [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
                [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
                [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
                [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
                [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
                [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
                [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
                [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
                [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
                [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
                [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
                [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
                [102, 255, 0], [92, 0, 255]]
        comfy_path = os.environ.get('COMFYUI_PATH')
        if comfy_path is None:
            comfy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        
        model_path = os.path.abspath(os.path.join(comfy_path, 'models'))
        processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_dinat_large", cache_dir=model_path).to("cuda")
        model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_dinat_large", cache_dir=model_path)
        images = []
        for i in range(len(image)):
            img = image[i]
            if isinstance(img, torch.Tensor):
                img = 255. * img.cpu().numpy()
            else:
                img = 255. * img 
            img = np.clip(img, 0, 255).astype(np.uint8)
            img = Image.fromarray(img)
            
            semantic_inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt").to("cuda")
            with torch.no_grad():
                semantic_outputs = model(**semantic_inputs)
            predicted_semantic_map = processor.post_process_semantic_segmentation(semantic_outputs, target_sizes=[image.size[::-1]])[0]
            predicted_semantic_map_np = predicted_semantic_map.cpu().numpy()
            palette = np.array(palette)
            # Map the tensor values to the palette
            color_image = palette[predicted_semantic_map_np]
            color_image = torch.from_numpy(color_image / 255.0).unsqueeze(0)  # Add batch dimension
            images.append(color_image)
        
        images = torch.cat(images, dim=0)
        return (images,)
