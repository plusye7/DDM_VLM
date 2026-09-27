import torch
import numpy as np
from transformers import AutoConfig, AutoModel, AutoTokenizer
from PIL import Image

from DDM.config import config_get, load_model_config
from DDM.models.Owl3.compact_module import (
    ensure_generation_mixin,
    install_siglip_hidden_states_compat,
    mplug_owl3_qwen2_compat,
)


CONFIG = load_model_config("Owl3")

class OWL3:

    _model = None
    _processor = None
    _tokenizer = None
    _device = None
    _initialized = False
    _model_path = config_get(CONFIG, "model.path")

    _chat_history = []
    _multi_turn = False

    # Initialize once.
    @staticmethod
    def initialize(multi_turn=False):
        if OWL3._initialized:
            return
    
        print("🚀 Loading OWL3 model...")
    
        loaded_model = None
        tokenizer = None
        processor = None
    
        model_config = AutoConfig.from_pretrained(
            OWL3._model_path,
            trust_remote_code=True,
        )
    
        model_config._attn_implementation = "sdpa"
    
    
        vision_config = getattr(model_config, "vision_config", None)
    
        if vision_config is None:
            raise AttributeError(
                "OWL3 config does not contain 'vision_config'."
            )
    
        vision_config._attn_implementation = "eager"
    
        tokenizer = AutoTokenizer.from_pretrained(
            OWL3._model_path,
            trust_remote_code=True,
        )
    
    
        with mplug_owl3_qwen2_compat():
            loaded_model = AutoModel.from_pretrained(
                OWL3._model_path,
                trust_remote_code=True,
                config=model_config,
                torch_dtype=torch.bfloat16,
            )
    
        actual_vision_model = getattr(
            loaded_model,
            "vision_model",
            None,
        )
    
        if actual_vision_model is None:
            raise AttributeError(
                "Loaded OWL3 model does not contain 'vision_model'."
            )
    
        actual_vision_config = getattr(
            actual_vision_model,
            "config",
            None,
        )
    
        if actual_vision_config is None:
            raise AttributeError(
                "OWL3 vision_model does not contain 'config'."
            )
    
        actual_vision_config._attn_implementation = "eager"
    
        install_siglip_hidden_states_compat(loaded_model)
    
    
        language_model = getattr(
            loaded_model,
            "language_model",
            None,
        )
    
        if language_model is None:
            raise AttributeError(
                "Loaded OWL3 model does not contain 'language_model'."
            )
    
        ensure_generation_mixin(language_model)
    
        loaded_model = loaded_model.eval().cuda()
    
        try:
            model_device = next(loaded_model.parameters()).device
        except StopIteration:
            model_device = torch.device("cuda:0")
    
        processor = loaded_model.init_processor(tokenizer)
    
        if processor is None:
            raise RuntimeError(
                "OWL3 init_processor() returned None."
            )
    
        OWL3._model = loaded_model
        OWL3._processor = processor
        OWL3._tokenizer = tokenizer
        OWL3._device = model_device
    
        OWL3._multi_turn = multi_turn
        OWL3._chat_history = []
        OWL3._initialized = True
    
        print("✅ OWL3 model loaded successfully. ")

    @staticmethod
    def reset_chat():
        OWL3._chat_history = []

    # Build the input prompt and image list.
    @staticmethod
    def _build_prompt_and_images(prompt_text, query_image,
                                 example_texts=None, example_images=None):
        """
        Build multimodal few-shot input:
        - query_image: query image
        - example_texts: example class text
        - example_images: example images
        """
        content = ""
        images = []

        # ---------- Add examples. ----------
        if example_images is not None and example_texts is not None:
            for i, (img, txt) in enumerate(zip(example_images, example_texts)):
                content += f"<|image|> \nExample {i+1}:\nCorrect classification result: {txt}\n\n"

                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)

                images.append(img)

        # ---------- Add the query image. ----------
        content += "Now classify the following image.\n<|image|> \n"
        images.append(query_image)
        content += prompt_text

        return content, images

    # Core prediction function matching the DeepSeek interface.
    @staticmethod
    def predict(prompt_text,
            image,
            example_texts=None,
            example_images=None,
            max_new_tokens=512):

        if not OWL3._initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        # ---------- Build the prompt and image list. ----------
        content, images = OWL3._build_prompt_and_images(
            prompt_text,
            image,
            example_texts,
            example_images
        )

        # ---------- Build the conversation in OWL3 format. ----------
        messages = []

        # If multi-turn mode is enabled.
        if OWL3._multi_turn and len(OWL3._chat_history) > 0:
            messages.extend(OWL3._chat_history)

        # Current turn.
        messages.append({
            "role": "user",
            "content": content
        })

        messages.append({
            "role": "assistant",
            "content": ""
        })

        # ---------- Process inputs. ----------
        inputs = OWL3._processor(messages, images=images, videos=None)
        inputs = {k: v.to(OWL3._device) if torch.is_tensor(v) else v for k, v in inputs.items()}

        # ---------- Generate text and collect logits for each step. ----------
        with torch.no_grad():
            # Run forward first to get logits.
            outputs = OWL3._model(
                **inputs,
                use_cache=False,
                return_dict=True
            )

            # Then generate text.
            inputs.update({
                'tokenizer': OWL3._tokenizer,
                'max_new_tokens': max_new_tokens,
                'decode_text': True,
                'use_cache': False,
            })

            generation_output = OWL3._model.generate(**inputs)

        # Get generated token IDs.
        generated_ids = None
        if isinstance(generation_output, torch.Tensor):
            generated_ids = generation_output[0].cpu().numpy()
        elif isinstance(generation_output, list):
            try:
                generated_ids = OWL3._tokenizer.encode(generation_output[0], add_special_tokens=False)
                generated_ids = np.array(generated_ids)
            except:
                generated_ids = np.array([])
        elif isinstance(generation_output, str):
            try:
                generated_ids = OWL3._tokenizer.encode(generation_output, add_special_tokens=False)
                generated_ids = np.array(generated_ids)
            except:
                generated_ids = np.array([])
        else:
            generated_ids = np.array([])

        # Collect logits for all generation steps.
        # Get logits for all tokens from the forward output.
        logits = outputs.logits  # [1, seq_len, vocab_size]

        # Convert to a NumPy array [seq_len, vocab_size]
        generation_logits = logits[0].float().cpu().numpy()

        # Decode the full output.
        raw_output = ""
        if generated_ids.size > 0:
            raw_output = OWL3._tokenizer.decode(generated_ids, skip_special_tokens=True)

        if OWL3._multi_turn:
            OWL3._chat_history.append({
                "role": "user",
                "content": prompt_text
            })

            OWL3._chat_history.append({
                "role": "assistant",
                "content": raw_output
            })

        return {
            "generation_logits": generation_logits,      # [seq_len, vocab_size] - logits for all input tokens
            "generated_token_ids": generated_ids,         # all generated token IDs
            "raw_output": raw_output,                     # fully decoded text
            "num_generated_tokens": len(generated_ids) if generated_ids.size > 0 else 0  # number of generated tokens
        }
