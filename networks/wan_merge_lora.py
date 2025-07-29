import argparse
import math
import os
import time
from typing import Any, Dict, Union

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from library.utils import setup_logging, str_to_dtype, MemoryEfficientSafeOpen, mem_eff_save_file

setup_logging()
import logging

logger = logging.getLogger(__name__)

import networks.lora_flux as lora_flux
from library import sai_model_spec, train_util


def load_state_dict(file_name, dtype):
    if os.path.splitext(file_name)[1] == ".safetensors":
        sd = load_file(file_name)
        metadata = train_util.load_metadata_from_safetensors(file_name)
    else:
        sd = torch.load(file_name, map_location="cpu")
        metadata = {}

    for key in list(sd.keys()):
        if type(sd[key]) == torch.Tensor:
            sd[key] = sd[key].to(dtype)

    return sd, metadata


def save_to_file(file_name, state_dict: Dict[str, Union[Any, torch.Tensor]], dtype, metadata, mem_eff_save=False):
    if dtype is not None:
        logger.info(f"converting to {dtype}...")
        for key in tqdm(list(state_dict.keys())):
            if type(state_dict[key]) == torch.Tensor and state_dict[key].dtype.is_floating_point:
                state_dict[key] = state_dict[key].to(dtype)

    logger.info(f"saving to: {file_name}")
    if mem_eff_save:
        mem_eff_save_file(state_dict, file_name, metadata=metadata)
    else:
        save_file(state_dict, file_name, metadata=metadata)


def merge_to_flux_model_extended(
    loading_device,
    working_device,
    flux_path: str,
    clip_l_path: str,
    t5xxl_path: str,
    models,
    ratios,
    merge_dtype,
    save_dtype,
    mem_eff_load_save=False,
    patch_embedding_model=None,
    replace_keys=None
):
    # create module map without loading state_dict
    lora_name_to_module_key = {}
    if flux_path is not None:
        logger.info(f"loading keys from Wan 2.1 model: {flux_path}")
        with safe_open(flux_path, framework="pt", device=loading_device) as flux_file:
            keys = list(flux_file.keys())
            for key in keys:
                if key.endswith(".weight") or key.endswith(".bias"):
                    module_name = ".".join(key.split(".")[:-1])
                    lora_name = lora_flux.LoRANetwork.LORA_PREFIX_FLUX + "_" + module_name.replace(".", "_")
                    lora_name_to_module_key[lora_name] = module_name
                elif key.endswith(".modulation"):
                    module_name = key
                    lora_name = lora_flux.LoRANetwork.LORA_PREFIX_FLUX + "_" + module_name.replace(".", "_")
                    lora_name_to_module_key[lora_name] = module_name
                

    flux_state_dict = {}

    if mem_eff_load_save:
        if flux_path is not None:
            with MemoryEfficientSafeOpen(flux_path) as flux_file:
                for key in tqdm(flux_file.keys()):
                    flux_state_dict[key] = flux_file.get_tensor(key).to(loading_device)


    else:
        if flux_path is not None:
            flux_state_dict = load_file(flux_path, device=loading_device)

    for model, ratio in zip(models, ratios):
        logger.info(f"loading: {model}")
        lora_sd, lora_metadata = load_state_dict(model, merge_dtype)

        # Check if this is an extended LoRA
        is_extended_lora = lora_metadata.get("ss_extended_lora", "false").lower() == "true"
        logger.info(f"Extended LoRA: {is_extended_lora}")

        logger.info(f"merging...")
        processed_keys = set()
        # if not len(lora_sd.keys()) == len(lora_name_to_module_key):
        #     import ipdb; ipdb.set_trace()
        #     print(f"number of keys in LoRA model: {len(lora_sd.keys())}")
        #     print(f"number of keys in Wan 2.1 model: {len(lora_name_to_module_key)}")
        #     print(f"non-matching keys: {set(lora_name_to_module_key.keys()) - set(lora_sd.keys())}")
        for key in tqdm(list(lora_sd.keys())):
            if key in processed_keys:
                continue
            # import ipdb; ipdb.set_trace()
            # Handle standard LoRA weights (SVD-based)
            if "lora_down" in key:
                lora_name = key[: key.rfind(".lora_down")]
                up_key = key.replace("lora_down", "lora_up")
                alpha_key = key[: key.index("lora_down")] + "alpha"

                if lora_name in lora_name_to_module_key:
                    module_weight_key = lora_name_to_module_key[lora_name] + ".weight"
                    state_dict = flux_state_dict

                else:
                    logger.warning(f"no module found for LoRA weight: {key}. Skipping...")
                    continue

                down_weight = lora_sd.pop(key)
                up_weight = lora_sd.pop(up_key)
                processed_keys.add(key)
                processed_keys.add(up_key)

                dim = down_weight.size()[0]
                alpha = lora_sd.pop(alpha_key, dim)
                if alpha_key in lora_sd:
                    processed_keys.add(alpha_key)
                scale = alpha / dim

                # W <- W + U * D
                weight = state_dict[module_weight_key]

                weight = weight.to(working_device, merge_dtype)
                up_weight = up_weight.to(working_device, merge_dtype)
                down_weight = down_weight.to(working_device, merge_dtype)

                if len(weight.size()) == 2:
                    # linear
                    weight = weight + ratio * (up_weight @ down_weight) * scale
                elif down_weight.size()[2:4] == (1, 1):
                    raise ValueError("conv2d 1x1 is not supported for Wan 2.1 model")
                    # conv2d 1x1
                    weight = (
                        weight
                        + ratio
                        * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                        * scale
                    )
                else:
                    raise ValueError("conv2d is not supported for Wan 2.1 model")
                    # conv2d 3x3
                    conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                    weight = weight + ratio * conved * scale

                state_dict[module_weight_key] = weight.to(loading_device, save_dtype)
                del up_weight
                del down_weight
                del weight

            # Handle extended LoRA weights (direct differences)
            elif is_extended_lora and (".diff" in key or ".diff_b" in key or ".modulation" in key):
                if ".diff_b" in key:
                    lora_name = key[: key.rfind(".diff_b")]
                    weight_suffix = ".bias"
                elif ".diff" in key:
                    lora_name = key[: key.rfind(".diff")]
                    # weight_suffix = ".weight"
                    weight_suffix = ""
                    if "norm" in key:
                        weight_suffix = ".weight"
                elif ".modulation" in key:
                    lora_name = key
                    weight_suffix = ""
                else:
                    raise ValueError(f"Unsupported key: {key}")

                if lora_name in lora_name_to_module_key:
                    # Check if the corresponding weight/bias key exists
                    potential_key = lora_name_to_module_key.get(lora_name)
                    if potential_key:
                        # Replace .weight with the appropriate suffix
                        # if potential_key.endswith(".weight"):
                        #     module_key = potential_key[:-7] + weight_suffix
                        # else:
                        #     module_key = potential_key + weight_suffix
                        module_key = potential_key + weight_suffix
                        
                        # Check if this key actually exists in the state dict
                        if module_key in flux_state_dict:
                            state_dict = flux_state_dict
                        else:
                            import ipdb; ipdb.set_trace()
                            logger.warning(f"Module key {module_key} not found in flux model. Skipping...")
                            continue
                    else:
                        logger.warning(f"no module found for extended LoRA weight: {key}. Skipping...")
                        continue

                else:
                    import ipdb; ipdb.set_trace()
                    logger.warning(f"no module found for extended LoRA weight: {key}. Skipping...")
                    continue

                diff_weight = lora_sd.pop(key)
                processed_keys.add(key)

                # Apply the difference directly
                weight = state_dict[module_key]
                weight = weight.to(working_device, merge_dtype)
                diff_weight = diff_weight.to(working_device, merge_dtype)

                # Direct addition of the difference
                weight = weight + ratio * diff_weight

                state_dict[module_key] = weight.to(loading_device, save_dtype)
                del diff_weight
                del weight

        # Check for any remaining unprocessed keys
        remaining_keys = set(lora_sd.keys()) - processed_keys
        if remaining_keys:
            logger.warning(f"Unused keys in LoRA model: {list(remaining_keys)}")
    # import ipdb; ipdb.set_trace()
    # if the key contains "model.diffusion_model.", remote it
    new_ckpt_state_dict = {}
    for key in flux_state_dict.keys():
        if "model.diffusion_model." in key:
            new_key = key.replace("model.diffusion_model.", "")
            new_ckpt_state_dict[new_key] = flux_state_dict[key]
        else:
            new_ckpt_state_dict[key] = flux_state_dict[key]
    
    if patch_embedding_model is not None:
        replace_keys_list = []
        patch_embedding_state_dict = load_file(patch_embedding_model, device=loading_device)
        for key, value in new_ckpt_state_dict.items():
            if key in patch_embedding_state_dict:
                if replace_keys is not None:
                    for kk in replace_keys:
                        if kk in key:
                            replace_keys_list.append(key)
                else:
                    if value.shape == patch_embedding_state_dict[key].shape:
                        continue
                    else:
                        replace_keys_list.append(key)
            else:
                raise ValueError(f"Key {key} not found in patch embedding model")

        for key in replace_keys_list:
            logger.info(f"Replacing {key} with {patch_embedding_state_dict[key].shape} from {new_ckpt_state_dict[key].shape}")
            new_ckpt_state_dict[key] = patch_embedding_state_dict[key]
            if key.endswith(".weight"):
                # replace bias
                bias_key = key[:-7] + ".bias"
                if bias_key in new_ckpt_state_dict:
                    new_ckpt_state_dict[bias_key] = patch_embedding_state_dict[bias_key]
                    logger.info(f"Replacing {bias_key} with {patch_embedding_state_dict[bias_key].shape} from {new_ckpt_state_dict[bias_key].shape}")
                else:
                    logger.warning(f"Bias key {bias_key} not found in new ckpt state dict. Skipping...")
    
    return new_ckpt_state_dict



def merge(args):
    if args.models is None:
        args.models = []
    if args.ratios is None:
        args.ratios = []

    assert len(args.models) == len(
        args.ratios
    ), "number of models must be equal to number of ratios"

    merge_dtype = str_to_dtype(args.precision)
    save_dtype = str_to_dtype(args.save_precision)
    if save_dtype is None:
        save_dtype = merge_dtype

    assert (
        args.save_to
    ), "save_to must be specified"
    dest_dir = os.path.dirname(args.save_to)
    if not os.path.exists(dest_dir):
        logger.info(f"creating directory: {dest_dir}")
        os.makedirs(dest_dir)

    if args.flux_model is not None:

        flux_state_dict = merge_to_flux_model_extended(
            args.loading_device,
            args.working_device,
            args.flux_model,
            args.clip_l,
            args.t5xxl,
            args.models,
            args.ratios,
            merge_dtype,
            save_dtype,
            args.mem_eff_load_save,
            args.patch_embedding_model,
            args.replace_keys
        )

        if args.no_metadata or (flux_state_dict is None or len(flux_state_dict) == 0):
            sai_metadata = None
        else:
            merged_from = sai_model_spec.build_merged_from([args.flux_model] + args.models)
            title = os.path.splitext(os.path.basename(args.save_to))[0]
            sai_metadata = sai_model_spec.build_metadata(
                None, False, False, False, False, False, time.time(), title=title, merged_from=merged_from, flux="dev"
            )

        if flux_state_dict is not None and len(flux_state_dict) > 0:
            logger.info(f"saving FLUX model to: {args.save_to}")
            save_to_file(args.save_to, flux_state_dict, save_dtype, sai_metadata, args.mem_eff_load_save)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        help="precision in saving, same to merging if omitted. supported types: "
        "float32, fp16, bf16, fp8 (same as fp8_e4m3fn), fp8_e4m3fn, fp8_e4m3fnuz, fp8_e5m2, fp8_e5m2fnuz",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float",
        help="precision in merging (float is recommended)",
    )
    parser.add_argument(
        "--flux_model",
        type=str,
        default=None,
        help="FLUX.1 model to load, merge LoRA models if omitted",
    )
    parser.add_argument(
        "--patch_embedding_model",
        type=str,
        default=None,
        help="patch embedding model to load, merge LoRA models if omitted",
    )
    parser.add_argument(
        "--clip_l",
        type=str,
        default=None,
        help="path to clip_l (*.sft or *.safetensors), should be float16",
    )
    parser.add_argument(
        "--t5xxl",
        type=str,
        default=None,
        help="path to t5xxl (*.sft or *.safetensors), should be float16",
    )
    parser.add_argument(
        "--mem_eff_load_save",
        action="store_true",
        help="use custom memory efficient load and save functions for FLUX.1 model",
    )
    parser.add_argument(
        "--loading_device",
        type=str,
        default="cpu",
        help="device to load FLUX.1 model. LoRA models are loaded on CPU",
    )
    parser.add_argument(
        "--working_device",
        type=str,
        default="cuda",
        help="device to work (merge). Merging LoRA models are done on CPU.",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        default=None,
        help="destination file name: safetensors file",
    )
    parser.add_argument(
        "--clip_l_save_to",
        type=str,
        default=None,
        help="destination file name for clip_l: safetensors file",
    )
    parser.add_argument(
        "--t5xxl_save_to",
        type=str,
        default=None,
        help="destination file name for t5xxl: safetensors file",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        help="LoRA models to merge: safetensors file",
    )
    parser.add_argument("--ratios", type=float, nargs="*", help="ratios for each model")
    parser.add_argument(
        "--no_metadata",
        action="store_true",
        help="do not save sai modelspec metadata (minimum ss_metadata for LoRA is saved)",
    )
    parser.add_argument(
        "--concat",
        action="store_true",
        help="concat lora instead of merge (The dim(rank) of the output LoRA is the sum of the input dims)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="shuffle lora weight.",
    )
    parser.add_argument(
        "--replace_keys",
        type=str,
        nargs="*",
        help="keys to replace in the flux model.",
        default=None
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    merge(args) 