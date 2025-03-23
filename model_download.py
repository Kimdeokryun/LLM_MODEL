import kagglehub

#kaggle API request

# Download latest version
path = kagglehub.model_download("google/gemma-3/pyTorch/gemma-3-27b-it")

print("Path to model files:", path)