import math
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig, AutoModel, AutoTokenizer

from DDM.config import config_get, load_model_config


CONFIG = load_model_config("Internvl3")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

def _env_flag(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

def _to_pil_image(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)!r}")

def _build_transform(input_size):
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize(
                (input_size, input_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

def _find_closest_aspect_ratio(
    aspect_ratio,
    target_ratios,
    width,
    height,
    image_size,
):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)

        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            target_area = image_size * image_size * ratio[0] * ratio[1]
            if area > 0.5 * target_area:
                best_ratio = ratio

    return best_ratio

def _dynamic_preprocess(
    image,
    min_num=1,
    max_num=1,
    image_size=448,
    use_thumbnail=True,
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        orig_width,
        orig_height,
        image_size,
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))

    processed_images = []
    for idx in range(blocks):
        box = (
            (idx % (target_width // image_size)) * image_size,
            (idx // (target_width // image_size)) * image_size,
            ((idx % (target_width // image_size)) + 1) * image_size,
            ((idx // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images

class InternVL3:
    _model = None
    _tokenizer = None
    _device = None
    _initialized = False

    _model_path = os.environ.get(
        "INTERNVL3_MODEL_PATH",
        config_get(CONFIG, "model.path"),
    )
    _image_size = int(
        os.environ.get(
            "INTERNVL3_IMAGE_SIZE",
            config_get(CONFIG, "model.image_size", 448),
        )
    )
    _max_tiles = int(
        os.environ.get(
            "INTERNVL3_MAX_TILES",
            config_get(CONFIG, "model.max_tiles", 1),
        )
    )
    _use_flash_attn = _env_flag(
        "INTERNVL3_USE_FLASH_ATTN",
        bool(config_get(CONFIG, "model.use_flash_attn", False)),
    )
    _load_in_8bit = _env_flag(
        "INTERNVL3_LOAD_IN_8BIT",
        bool(config_get(CONFIG, "model.load_in_8bit", False)),
    )
    _device_map_mode = os.environ.get(
        "INTERNVL3_DEVICE_MAP",
        str(config_get(CONFIG, "model.device_map", "")),
    ).strip()

    _chat_history = []
    _multi_turn = False

    @staticmethod
    def initialize(multi_turn=False):
        if InternVL3._initialized:
            return

        InternVL3._multi_turn = multi_turn
        InternVL3._chat_history = []

        print("Loading InternVL3-8B model...")

        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }

        if InternVL3._use_flash_attn:
            load_kwargs["use_flash_attn"] = True
        if InternVL3._load_in_8bit:
            load_kwargs["load_in_8bit"] = True

        device_map = InternVL3._resolve_device_map()
        if device_map is not None:
            load_kwargs["device_map"] = device_map

        InternVL3._model = AutoModel.from_pretrained(
            InternVL3._model_path,
            **load_kwargs,
        ).eval()

        if device_map is None and not InternVL3._load_in_8bit:
            if torch.cuda.is_available():
                InternVL3._model = InternVL3._model.cuda()
            else:
                InternVL3._model = InternVL3._model.to("cpu")

        InternVL3._tokenizer = AutoTokenizer.from_pretrained(
            InternVL3._model_path,
            trust_remote_code=True,
            use_fast=False,
        )
        InternVL3._device = InternVL3._infer_input_device()
        InternVL3._initialized = True

        print("InternVL3-8B model loaded successfully.")

    @staticmethod
    def reset_chat():
        InternVL3._chat_history = []

    @staticmethod
    def _resolve_device_map():
        mode = InternVL3._device_map_mode
        if not mode:
            return None
        if mode.lower() == "split":
            return InternVL3._split_model()
        return mode

    @staticmethod
    def _split_model():
        world_size = torch.cuda.device_count()
        if world_size <= 1:
            return None

        config = AutoConfig.from_pretrained(
            InternVL3._model_path,
            trust_remote_code=True,
        )
        num_layers = config.llm_config.num_hidden_layers

        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)

        device_map = {}
        layer_cnt = 0
        for gpu_idx, layer_count in enumerate(num_layers_per_gpu):
            for _ in range(layer_count):
                if layer_cnt >= num_layers:
                    break
                device_map[f"language_model.model.layers.{layer_cnt}"] = gpu_idx
                layer_cnt += 1

        device_map["vision_model"] = 0
        device_map["mlp1"] = 0
        device_map["language_model.model.embed_tokens"] = 0
        device_map["language_model.model.norm"] = 0
        device_map["language_model.model.rotary_emb"] = 0
        device_map["language_model.lm_head"] = 0
        device_map[f"language_model.model.layers.{num_layers - 1}"] = 0
        return device_map

    @staticmethod
    def _infer_input_device():
        if hasattr(InternVL3._model, "device"):
            return torch.device(InternVL3._model.device)

        for _, param in InternVL3._model.named_parameters():
            if param.device.type != "meta":
                return param.device

        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _image_to_pixel_values(image):
        image = _to_pil_image(image)
        transform = _build_transform(InternVL3._image_size)
        tiles = _dynamic_preprocess(
            image,
            image_size=InternVL3._image_size,
            max_num=InternVL3._max_tiles,
            use_thumbnail=True,
        )
        pixel_values = [transform(tile) for tile in tiles]
        return torch.stack(pixel_values)

    @staticmethod
    def _build_question_and_pixel_values(
        prompt_text,
        query_image,
        example_texts=None,
        example_images=None,
    ):
        parts = []
        images = []

        if example_images is not None and example_texts is not None:
            if len(example_images) != len(example_texts):
                raise ValueError("example_images and example_texts must have same length")

            for idx, (img, text) in enumerate(zip(example_images, example_texts)):
                parts.append(f"Image-{idx + 1}: <image>\nExample {idx + 1}:\n{text}")
                images.append(img)

        query_idx = len(images) + 1
        if images:
            parts.append(
                f"Image-{query_idx}: <image>\n"
                f"Now classify the following image.\n{prompt_text}"
            )
        else:
            parts.append(f"<image>\n{prompt_text}")
        images.append(query_image)

        pixel_chunks = []
        num_patches_list = []
        for img in images:
            pixel_values = InternVL3._image_to_pixel_values(img)
            pixel_chunks.append(pixel_values)
            num_patches_list.append(int(pixel_values.shape[0]))

        pixel_values = torch.cat(pixel_chunks, dim=0)
        pixel_values = pixel_values.to(torch.bfloat16).to(InternVL3._device)

        return "\n\n".join(parts), pixel_values, num_patches_list

    @staticmethod
    def _get_conv_template():
        module = sys.modules.get(InternVL3._model.__class__.__module__)
        get_conv_template = getattr(module, "get_conv_template", None)
        if get_conv_template is None:
            raise RuntimeError("InternVL3 conversation template helper is unavailable.")

        template = get_conv_template(InternVL3._model.template)
        template.system_message = InternVL3._model.system_message
        return template

    @staticmethod
    def _generate_with_scores(
        question,
        pixel_values,
        num_patches_list,
        max_new_tokens,
    ):
        tokenizer = InternVL3._tokenizer
        model = InternVL3._model
        template = InternVL3._get_conv_template()

        model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

        if InternVL3._multi_turn:
            history = InternVL3._chat_history
        else:
            history = []

        for old_question, old_answer in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)

        query = template.get_prompt()
        for num_patches in num_patches_list:
            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches
                + IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(InternVL3._device)
        attention_mask = model_inputs["attention_mask"].to(InternVL3._device)

        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "eos_token_id": eos_token_id,
            "pad_token_id": tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
        }

        with torch.no_grad():
            generation_output = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_config,
            )

        sequences = generation_output.sequences
        generated_ids = sequences[0].detach().cpu().numpy()

        raw_output = tokenizer.batch_decode(
            sequences,
            skip_special_tokens=True,
        )[0]
        raw_output = raw_output.split(template.sep.strip())[0].strip()

        scores = getattr(generation_output, "scores", None) or []
        generation_logits = (
            np.asarray(
                [score[0].detach().float().cpu().numpy() for score in scores],
                dtype=np.float32,
            )
            if len(scores) > 0
            else np.array([])
        )

        return raw_output, generated_ids, generation_logits

    @staticmethod
    def predict(
        prompt_text,
        image,
        example_texts=None,
        example_images=None,
        max_new_tokens=512,
    ):
        if not InternVL3._initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        question, pixel_values, num_patches_list = (
            InternVL3._build_question_and_pixel_values(
                prompt_text,
                image,
                example_texts,
                example_images,
            )
        )

        raw_output, generated_ids, generation_logits = InternVL3._generate_with_scores(
            question,
            pixel_values,
            num_patches_list,
            max_new_tokens,
        )

        if InternVL3._multi_turn:
            InternVL3._chat_history.append((question, raw_output))

        return {
            "generation_logits": generation_logits,
            "generated_token_ids": generated_ids,
            "raw_output": raw_output,
            "num_generated_tokens": len(generated_ids),
        }
