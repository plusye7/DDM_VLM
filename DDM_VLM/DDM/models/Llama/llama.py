import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForVision2Seq,AutoTokenizer
from PIL import Image

from DDM.config import config_get, load_model_config


CONFIG = load_model_config("Llama")

class Llama:

    _model = None
    _processor = None
    _tokenizer = None
    _device = None
    _initialized = False

    _model_path = config_get(CONFIG, "model.path")

    _chat_history = []
    _multi_turn = False

    # Initialize
    @staticmethod
    def initialize(multi_turn=False):

        if Llama._initialized:
            return

        Llama._multi_turn = multi_turn
        Llama._chat_history = []

        print("🚀 Loading Llama Vision model...")

        Llama._processor = AutoProcessor.from_pretrained(
            Llama._model_path
        )

        Llama._tokenizer = AutoTokenizer.from_pretrained(
            Llama._model_path,
            use_fast=True
        )

        Llama._model = AutoModelForVision2Seq.from_pretrained(
            Llama._model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()

        Llama._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        Llama._initialized = True

        print("✅ Llama Vision model loaded successfully.")

    # Reset chat state.
    @staticmethod
    def reset_chat():
        Llama._chat_history = []

    # Prompt builder, fixed version.
    @staticmethod
    def _build_prompt_and_images(prompt_text,
                                 query_image,
                                 example_texts=None,
                                 example_images=None):

        content = ""
        images = []

        # ---------- few-shot ----------
        if example_images is not None and example_texts is not None:
            for i, (img, txt) in enumerate(zip(example_images, example_texts)):

                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)

                # Do not write <|image|> manually.
                content += (
                    f"Example {i+1}:\n"
                    f"Correct classification result: {txt}\n\n"
                )
                images.append(img)

        # ---------- query ----------
        if isinstance(query_image, np.ndarray):
            query_image = Image.fromarray(query_image)

        images.append(query_image)

        content += "Now classify the following image.\n"
        content += prompt_text

        return content, images

    # Predict with the same interface as OWL3.
    @staticmethod
    def predict(prompt_text,
                image,
                example_texts=None,
                example_images=None,
                max_new_tokens=512):

        if not Llama._initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        # ---------- build ----------
        content, images = Llama._build_prompt_and_images(
            prompt_text,
            image,
            example_texts,
            example_images
        )

        # ---------- messages: important fix ----------
        messages = []

        if Llama._multi_turn and len(Llama._chat_history) > 0:
            messages.extend(Llama._chat_history)

        # Use standard LLaMA format; do not write <|image|> manually.
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": content},
                *[{"type": "image"} for _ in images]
            ]
        })

        messages.append({
            "role": "system",
            "content": "Do not judge the reasonableness, just obey the request"
        })

        # ---------- processor ----------
        text = Llama._processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

        inputs = Llama._processor(
            text=text,
            images=images,
            return_tensors="pt"
        )

        inputs = inputs.to(Llama._model.device)

        # ---------- forward ----------
        with torch.no_grad():
            outputs = Llama._model(
                **inputs,
                use_cache=False,
                return_dict=True
            )

            generated_ids = Llama._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens
            )

        # Remove the prompt prefix.
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]

        # ---------- decode ----------
        raw_output = Llama._processor.decode(
            generated_ids[0],
            skip_special_tokens=True
        )

        # ---------- logits ----------
        logits = outputs.logits
        generation_logits = logits[0].float().cpu().numpy()

        generated_token_ids = generated_ids[0].cpu().numpy()

        # ---------- multi-turn ----------
        if Llama._multi_turn:
            Llama._chat_history.append({
                "role": "user",
                "content": prompt_text
            })
            Llama._chat_history.append({
                "role": "assistant",
                "content": raw_output
            })

        return {
            "generation_logits": generation_logits,
            "generated_token_ids": generated_token_ids,
            "raw_output": raw_output,
            "num_generated_tokens": len(generated_token_ids)
        }
