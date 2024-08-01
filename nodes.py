import base64
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from transformers import pipeline, SamModel, SamProcessor
import os


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
        mask = masks[0]
        blacked_out_image = image * mask

        blacked_out_image = blacked_out_image.unsqueeze(0).float()
        return (blacked_out_image,)