!pip install -q -U immutabledict sentencepiece 
!git clone https://github.com/google/gemma_pytorch.git

import contextlib
import torch
import os
import kagglehub

import sys 
sys.path.append("/kaggle/working/gemma_pytorch/") 

from gemma.config import get_model_config
from gemma.gemma3_model import Gemma3ForMultimodalLM

# Choose variant and machine type
VARIANT = '27b'
MACHINE_TYPE = 'cpu'
OUTPUT_LEN = 20
METHOD = 'it'

weights_dir = kagglehub.model_download(f"google/gemma-3/pytorch/gemma-3-{VARIANT}-{METHOD}/1")
tokenizer_path = os.path.join(weights_dir, 'tokenizer.model')
ckpt_path = os.path.join(weights_dir, f'model.ckpt')

# Set up model config.
model_config = get_model_config(VARIANT)
model_config.dtype = "float32" if MACHINE_TYPE == "cpu" else "float16"
model_config.tokenizer = tokenizer_path

@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)

# Instantiate the model and load the weights.
device = torch.device(MACHINE_TYPE)
with _set_default_tensor_type(model_config.get_dtype()):
    model = Gemma3ForMultimodalLM(model_config)
    model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])        
    model = model.to(device).eval()

# Chat templates

USER_CHAT_TEMPLATE = "<start_of_turn>user\n{prompt}<end_of_turn>\n"
MODEL_CHAT_TEMPLATE = "<start_of_turn>model\n{prompt}<end_of_turn>\n"

model.generate([[
    USER_CHAT_TEMPLATE.format(prompt="What is a good place for travel in the US?"),
    MODEL_CHAT_TEMPLATE.format(prompt="California."),
    USER_CHAT_TEMPLATE.format(prompt="What can I do in California?"),
    "<start_of_turn>model\n",
    ]], 
    device, 
    output_len=OUTPUT_LEN
)

# With Image Input
def read_image(url):
    import io
    import requests
    import PIL

    contents = io.BytesIO(requests.get(url).content)
    return PIL.Image.open(contents)

image_url = 'https://storage.googleapis.com/keras-cv/models/paligemma/cow_beach_1.png'
image = read_image(image_url)

model.generate(
    [['<start_of_turn>user\n', image, 'What animal is in this image?<end_of_turn>\n', '<start_of_turn>model\n']],
    device=device,
    output_len=OUTPUT_LEN,
)
