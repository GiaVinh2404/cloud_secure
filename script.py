import pandas as pd
import torch


if torch.cuda.is_available():
    print("GPU detected:", torch.cuda.get_device_name(0))
else:
    print("No GPU detected. Using CPU.")
