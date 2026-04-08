"""
echos/model_loader.py
Loads the base model with the requested quantization mode onto the correct GPU.
Supports: fp4 (NF4 bitsandbytes), int8, bf16, fp16, fp32.
"""

from __future__ import annotations
import logging
from typing import Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from config import ECHOSConfig, HardwareConfig, QuantMode

logger = logging.getLogger(__name__)


def build_bnb_config(hw: HardwareConfig) -> BitsAndBytesConfig | None:
    if hw.quant_mode == "fp4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, hw.bnb_4bit_compute_dtype.replace("fp", "float").replace("bf", "bfloat")),
            bnb_4bit_use_double_quant=hw.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=hw.bnb_4bit_quant_type,
        )
    if hw.quant_mode == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_base_model(cfg: ECHOSConfig) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads model + tokenizer.
    Base model goes on cfg.hardware.base_model_device.
    """
    hw = cfg.hardware
    logger.info(f"Loading '{cfg.model_name}' | quant={hw.quant_mode} | device={hw.base_model_device}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = build_bnb_config(hw)

    load_kwargs: dict = dict(
        trust_remote_code=True,
        torch_dtype=hw.torch_dtype(),
    )

    if bnb_config is not None:
        load_kwargs["quantization_config"] = bnb_config
        # bitsandbytes manages device placement
        load_kwargs["device_map"] = {"": hw.base_model_device}
    else:
        load_kwargs["device_map"] = {"": hw.base_model_device}

    if hw.max_memory is not None:
        load_kwargs["max_memory"] = hw.max_memory

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **load_kwargs)
    model.eval()

    logger.info(
        f"Model loaded | params={sum(p.numel() for p in model.parameters()) / 1e9:.2f}B | "
        f"dtype={next(model.parameters()).dtype}"
    )
    return model, tokenizer
