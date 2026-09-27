from PIL import Image

from DDM.config import config_get, load_model_config
from DDM.models.Qwen.Qwen import Qwen

CONFIG = load_model_config("Qwen")

# Initialize with multi-turn mode.
Qwen.initialize(multi_turn=True)

# Load the configured example image.
image = Image.open(config_get(CONFIG, "examples.image_path"))

# First round.
prompt1 = "What is shown in this image? Answer briefly."

result1 = Qwen.predict(prompt1, image)

print("Round 1:")
print(result1["raw_output"])
print()

# Second round, depends on the previous round.
prompt2 = "Why do you think so?"

result2 = Qwen.predict(prompt2, image)

print("Round 2:")
print(result2["raw_output"])
print()

# Third round, continue follow-up.
prompt3 = "Explain it in more detail."

result3 = Qwen.predict(prompt3, image)

print("Round 3:")
print(result3["raw_output"])
print()
