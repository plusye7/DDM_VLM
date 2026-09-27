from PIL import Image

from DDM.config import config_get, load_model_config
from DDM.models.Owl3.Owl3 import OWL3

CONFIG = load_model_config("Owl3")

# Initialize with multi-turn mode.
OWL3.initialize(multi_turn=True)

# Load the configured example image.
image = Image.open(config_get(CONFIG, "examples.image_path"))

# First round.
prompt1 = "What is shown in this image? Answer briefly."

result1 = OWL3.predict(prompt1, image)

print("Round 1:")
print(result1)
