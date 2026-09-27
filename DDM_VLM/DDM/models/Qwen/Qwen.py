import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
import copy

from DDM.config import config_get, load_model_config


CONFIG = load_model_config("Qwen")

class Qwen:

    _llm = None
    _processor = None
    _tokenizer = None
    _initialized = False
    _model_path = config_get(CONFIG, "model.path")

    _multi_turn = False
    _chat_messages = []

    # Initialize
    @staticmethod
    def initialize(multi_turn=False):
        if Qwen._initialized:
            return

        print("🚀 Loading Qwen2.5-VL with vLLM...")

        Qwen._llm = LLM(
            model=Qwen._model_path,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.75,
            max_model_len=4096,
            enforce_eager=True
        )

        Qwen._processor = AutoProcessor.from_pretrained(Qwen._model_path)
        Qwen._tokenizer = Qwen._processor.tokenizer

        Qwen._multi_turn = multi_turn

        Qwen._chat_messages = []

        Qwen._initialized = True
        print("✅ Qwen2.5-VL loaded successfully with vLLM.")

    # Reset chat state.
    @staticmethod
    def reset_chat():
        Qwen._chat_messages = []

    # Build the prompt and image list.
    @staticmethod
    def _build_prompt_and_images(prompt_text, query_image,
                                 example_texts=None, example_images=None):

        content = ""
        images = []

        if example_images is not None and example_texts is not None:
            for i, (img, txt) in enumerate(zip(example_images, example_texts)):
                content += f"Example {i+1}:\n{txt}\n\n"
                images.append(img)

        content += "Now classify the following image.\n"
        content += prompt_text

        images.append(query_image)

        return content, images

    @staticmethod
    def predict(prompt_text,
                image,
                example_texts=None,
                example_images=None,
                max_new_tokens=512):

        if not Qwen._initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        # Initialize history
        if Qwen._multi_turn and len(Qwen._chat_messages) > 0:
            messages = copy.deepcopy(Qwen._chat_messages)
            images = []
        else:
            messages = []
            images = []

        # Append a new user and assistant turn.
        messages.extend([
            {
                "role": "user",
                "content": []
            },
            {
                "role": "assistant",
                "content": []
            }
        ])

        user_msg = messages[-2]

        # few-shot examples
        if example_images is not None and example_texts is not None:

            if len(example_images) != len(example_texts):
                raise ValueError("example_images and example_texts must have same length")

            for img, txt in zip(example_images, example_texts):

                user_msg["content"].append({
                    "type": "image"
                })

                user_msg["content"].append({
                    "type": "text",
                    "text": txt
                })

                images.append(img)

        # query image and prompt
        user_msg["content"].append({
            "type": "image"
        })

        user_msg["content"].append({
            "type": "text",
            "text": prompt_text
        })

        images.append(image)

        # Build the prompt.
        text = Qwen._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Debug: validate image token count.
        image_tokens = 0

        for m in messages:
            content = m.get("content", [])

            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "image":
                        image_tokens += 1

        if image_tokens != len(images):
            raise RuntimeError(
                f"Image token mismatch: {image_tokens} tokens vs {len(images)} images"
            )

        # Sampling parameters.
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0,
            logprobs=20
        )

        # vLLM inference.
        outputs = Qwen._llm.generate(
            {
                "prompt": text,
                "multi_modal_data": {
                    "image": images
                }
            },
            sampling_params,
            use_tqdm=False,
        )

        output = outputs[0].outputs[0]

        # token ids
        generated_ids = np.array(output.token_ids)

        # Text output.
        raw_output = output.text

        # logprobs
        generation_logprobs = []

        for step_logprobs in output.logprobs:

            token_logprob_dict = {
                token_id: obj.logprob
                for token_id, obj in step_logprobs.items()
            }

            generation_logprobs.append(token_logprob_dict)

        # Update history.
        if Qwen._multi_turn:

            # Save text only, not images.
            user_history = {
                "role": "user",
                "content": []
            }

            for c in user_msg["content"]:
                if c["type"] == "text":
                    user_history["content"].append(c)

            Qwen._chat_messages.append(user_history)

            Qwen._chat_messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": raw_output
                    }
                ]
            })

        # Return the result.
        return {
            "generation_logprobs": generation_logprobs,
            "generated_token_ids": generated_ids,
            "raw_output": raw_output,
            "num_generated_tokens": len(generated_ids)
        }
