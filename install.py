os.system("pip3 install natten==0.17.1+torch240cu124 -f https://shi-labs.com/natten/wheels/")
os.system("pip3 install -r requirements.txt")

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
import os

def main():    
  # set HF_HOME env var
  comfy_path = os.environ.get('COMFYUI_PATH')
  if comfy_path is None:
      comfy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
  
  model_path = os.path.abspath(os.path.join(comfy_path, 'models'))
  print(f"Set HF_HOME to {model_path}")
  os.environ["HF_HOME"] = model_path
  if not os.path.exists(model_path):
    os.makedirs(model_path)

  OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_dinat_large", cache_dir=model_path)
  OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_dinat_large", cache_dir=model_path)
if __name__ == "__main__":
    main()

# pip3 install natten==0.17.1+torch240cu124 -f https://shi-labs.com/natten/wheels/
