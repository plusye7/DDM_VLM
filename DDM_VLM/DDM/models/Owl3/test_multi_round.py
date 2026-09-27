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
print(result1["raw_output"])
print()

# Second round, depends on the previous round.
prompt2 = "Why do you think so?"

result2 = OWL3.predict(prompt2, image)

print("Round 2:")
print(result2["raw_output"])
print()

# Third round, continue follow-up.
prompt3 = "Explain it in more detail."

result3 = OWL3.predict(prompt3, image)

print("Round 3:")
print(result3["raw_output"])
print()
