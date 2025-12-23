#!/usr/bin/env python3

import gc
import math
import os
import sys
import threading
import time
import warnings
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
import yaml
from numpy.typing import NDArray
from torch import nn

from sub.config import (EVAL_ITERS, LEARNING_RATE, LR_DECAY_ITERS, MIN_LR,
                        N_LAYERS_NODES, WARMUP_ITERS)
from sub.model import Config
from sub.utils.data_loader import get_batch

VERB = False


def get_obj_size(obj):
    """
    Get actual size of python object in memory (in bytes)
    """
    marked = {id(obj)}
    obj_q = [obj]
    sz = 0

    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object referred to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {
            o_id: o
            for o_id, o in all_refr
            if o_id not in marked and not isinstance(o, type)
        }

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return sz


@torch.no_grad()  # Tell the program not to evaluate the gradients (no BP)
def estimate_loss(
    model: nn.Module,
    train: Union[torch.Tensor, NDArray],
    val: Union[torch.Tensor, NDArray],
    batch_size: int,
    device: str,
    *args,
    **kwargs,
) -> Dict[str, float]:
    """
    Evaluate the mean loss over a fixed number of iterations during training.
    This allows to remove possible noise and provide more meaningful
    results.

    Args:
        model: the model on which to measure the loss
        train: training data set (tensor)
        val: validation data set (tensor)

    Returns:
        Dict containing the keys:
            "train": mean loss over EVAL_ITERS iterations for training set
            "val": mean loss over EVAL_ITERS iterations for validation set
    """
    ctx = kwargs.get("ctx", nullcontext())

    out = {}
    dss = {
        "train": train,
        "val": val,
    }
    # Set model to evaluation mode
    model.eval()
    for split in dss.keys():
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(dss[split], batch_size, device, model.config)
            with ctx:
                logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1
                )
            losses[k] = loss.item()
        out[split] = losses.mean()
    # Re-set the model to training mode
    model.train()
    return out


def get_lr(
    it,
    lr: float = LEARNING_RATE,
    min_lr: float = MIN_LR,
    warmup_it: int = WARMUP_ITERS,
    lr_decay_it: int = LR_DECAY_ITERS,
):
    """
    Evaluate learning rate for decayed LR.
    """
    # 1) linear warmup for warmup_iters steps
    if it < warmup_it:
        return lr * it / warmup_it
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_it:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_it) / (lr_decay_it - warmup_it)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (lr - min_lr)


def loading_bar(
    current_iter: int,
    tot_iter: int,
    n_chars: int = 10,
    ch: str = "=",
    n_ch: str = " ",
) -> str:
    """
    loading_bar
    ---
    Produce a loading bar string to be printed.

    Args:
        current_iter: current iteration, will determine the position
            of the current bar
        tot_iter: total number of iterations to be performed
        n_chars: total length of the loading bar in characters
        ch: character that makes up the loading bar (default: =)
        n_ch: character that makes up the remaining part of the bar
            (default: blankspace)

    Returns:
        string containing the loading bar for the current iteration
    """
    n_elem = int(current_iter * n_chars / tot_iter)
    prog = str("".join([ch] * n_elem))
    n_prog = str("".join([n_ch] * (n_chars - n_elem - 1)))
    return "[" + prog + n_prog + "]"


def waiting_animation(text: str, stopping: threading.Event):
    steps = ["⠴", "⠦", "⠇", "⠋", "⠙", "⠸"]
    stopping.clear()
    ind = 0
    while not stopping.is_set():
        print(text + f" {steps[ind]}", end="\r")
        ind += 1
        ind %= len(steps)
        time.sleep(0.5)
    print("")


def remove_prefix(text: str, prefix: str) -> str:
    """
    Remove the specified prefix from the given string.
    NOTE: starting Python 3.9, use text.removeprefix(prefix);
    """
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def find_eot(
    tokens: torch.Tensor,
    stop_tokens: Tuple[List[int], ...] = (),
    prompt_length: int = 0,
) -> torch.Tensor:
    """
    Return the sequence of tokens until the stopping tokens are found.
    The function finds the first EOS sequence starting from `prompt_length` (default 0)
    onwards.
    It will return the tensor truncated at the first EOS sequence after the prompt.

    Args:
        tokens: output of the LLM
        stop_tokens: tuple containing lists of the IDs representing the EOS
        prompt_length: optional prompt length
    """
    tok_lst = tokens.view(-1, 1).squeeze().tolist()
    assert (
        len(tok_lst) >= prompt_length
    ), "Prompt length must be longer than the provided tensor"
    start_ind = prompt_length + max([len(st) for st in stop_tokens])  # Skip prompt
    for i in range(start_ind, len(tok_lst)):
        if any(
            all(a == b for a, b in zip(tok_lst[i - len(st) : i], st))
            for st in stop_tokens
        ):
            return tokens[:, :i]
    return tokens


def detect_stop_tokens(
    tokens: torch.Tensor, stop_tokens: Tuple[List[int], ...] = ()
) -> bool:
    """
    Will return True if `tokens` terminates with one of the sequences defined in
    `stop_tokens`.
    """
    tok_lst = tokens.view(-1, 1).squeeze().tolist()
    return any(
        all(a == b for a, b in zip(tok_lst[-len(st) :], st)) for st in stop_tokens
    )


def detect_complete_answer(text: str, max_sentences: int = 2) -> bool:
    """
    Detect if the generated text contains a complete answer.
    Returns True if we should stop generation.
    
    Stops when:
    - More than max_sentences complete sentences have been generated
    - Repetitive sentence patterns detected (same structure repeated)
    - Repeated 5+ word phrases detected (strong indicator of looping)
    - Model starts generating a new "Question:" (repeating the format)
    
    Args:
        text: the generated text so far
        max_sentences: maximum number of sentences before stopping
    
    Returns:
        True if generation should stop
    """
    import re
    from collections import Counter

    text_lower = text.lower()
    
    # Also stop if we see multiple newlines (loop indicator)
    if '\n\n' in text:
        return True
    
    # Split into sentences (handle both . and newlines as separators)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|\n+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) > max_sentences:
        return True
    
    # 1. Detect repetitive sentence structure patterns
    # This catches things like "The capital of X is Y. The capital of Z is W."
    pattern_count = {}
    for sent in sentences:
        # Normalize: replace proper nouns with placeholder
        pattern = re.sub(r'\b[A-Z][a-z]+\b', '*', sent)
        pattern_count[pattern] = pattern_count.get(pattern, 0) + 1
        if pattern_count[pattern] > 1:
            return True
    
    # 2. Detect repeated 5-grams (very specific - strong indicator of looping)
    words = text_lower.split()
    if len(words) >= 10:
        fivegrams = [' '.join(words[i:i+5]) for i in range(len(words) - 4)]
        fivegram_counts = Counter(fivegrams)
        for phrase, count in fivegram_counts.items():
            if count > 1:
                return True
    
    return False


def truncate_to_complete_answer(text: str, max_sentences: int = 3) -> str:
    """
    Truncate text to remove incomplete sentences at the end.
    Keeps the full text up to the last complete sentence (ending with . ! or ?).
    
    Args:
        text: the generated text (may include prompt)
        max_sentences: not used currently, kept for API compatibility
    
    Returns:
        Text truncated at last complete sentence
    """
    import re
    
    text = text.strip()
    if not text:
        return text
    
    # If text already ends with sentence-ending punctuation, return as-is
    if text[-1] in '.!?':
        return text
    
    # Find the last occurrence of sentence-ending punctuation
    # and truncate there
    match = re.search(r'^(.*[.!?])', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # No complete sentence found, return original
    return text


def format_output(text: str):
    """
    Display the generated text correctly;

    This requires to isolate the <|user|> and <|assistant|> elements to isolate the
    specific things said by each.


    Maybe format with color??
    """
    pass


# def split_parameters(
#     model_params: Dict[str, Any], n_nodes: int
# ) -> Tuple[Dict[str, Any], Dict[str, int]]:
#     """
#     Split the model parameters (contained in a state dict) among the different
#     available nodes.
#     The model structure is that of LitGPT (https://github.com/Lightning-AI/litgpt).

#     The number of nodes should be at least 2 (starter and finisher).

#     The parameters are divided as such:
#         - Starter: token embedding,
#             N_LAYERS_STARTxTransformer Layers + final layer norm and linear layer
#         - Secondary: N_LAYERS_INTERMxTransformer Layer

#     Args:
#         model_params: complete model parameters (state dict)
#         n_nodes: number of nodes among which to divide the parameters; must be
#         greater or equal to 2 (at least starter and finisher)

#     Returns:
#         dict containing the following k-v pairs:
#             "starter": dict with the starter state dict
#             "secondary": list containing the intermediate state dicts
#         Layers information (n. layers per node)
#     """
#     assert n_nodes >= 2, "There must be at least 2 nodes in the network"

#     # Set up some parameters - they are used to gather the relevant keys
#     base_name_transformer = "transformer"  # Name of the ModuleDict in GPT
#     base_name_starter = "transformer"
#     base_name_secondary = "transformer"
#     tok_emb = "wte"
#     layer_name = "h"
#     transformer_last = f"{base_name_transformer}.ln_f"
#     output_layer = "lm_head"  # outside transformer now

#     # Count the number of detected transformer layers and check consistency
#     layer_keys = [
#         k
#         for k in model_params.keys()
#         if k.startswith(f"{base_name_transformer}.{layer_name}")
#     ]
#     layers_unique = list(set([".".join(k.split(".")[:3]) for k in layer_keys]))
#     n_layers_model = len(layers_unique)
#     if VERB:
#         print(f"Number of transformer layers found in the model: {n_layers_model}")

#     layers_info = {}
#     n_layers_start = N_LAYERS_NODES[n_nodes][n_layers_model]["N_LAYERS_START"]
#     layers_info["N_LAYERS_START"] = n_layers_start
#     n_layers_secondary = N_LAYERS_NODES[n_nodes][n_layers_model]["N_LAYERS_SECONDARY"]
#     layers_info["N_LAYERS_SECONDARY"] = n_layers_secondary

#     if VERB:
#         print(f"Number of layers - starter node: {n_layers_start}")
#         print(
#             f"Number of layers - secondary node{'s' if n_layers_secondary > 1 else ''}: {n_layers_secondary}"
#         )

#     out_chunks = {}

#     # 1. Select params for Starter
#     out_chunks["starter"] = {}
#     out_chunks["starter"][f"{base_name_starter}.{tok_emb}.weight"] = model_params.pop(
#         f"{base_name_transformer}.{tok_emb}.weight"
#     )
#     if f"{base_name_transformer}.{tok_emb}.bias" in model_params.keys():
#         out_chunks["starter"][f"{base_name_starter}.{tok_emb}.bias"] = model_params.pop(
#             f"{base_name_transformer}.{tok_emb}.bias"
#         )

#     # Starter transformer layers
#     # Complicated pythonic list call to select the correct keys to be transferred to the
#     # starter node
#     # As reference, the keys for the layers all start with:
#     #               transformer.h.<layer_ind>.[...]
#     # so we need to select the correct layer indices
#     valid_layer_ind = list(range(0, n_layers_start))
#     relevant_keys = [  # Keys of the original model that will be copied
#         k
#         for k in list(model_params.keys())
#         if (
#             k.startswith(f"{base_name_transformer}.{layer_name}")
#             and int(k.split(".")[2]) in valid_layer_ind
#         )
#     ]

#     for k_orig in relevant_keys:
#         ind_layer = int(k_orig.split(".")[2])
#         ind_layer_chunk = ind_layer  # Starter layers will have the same index

#         prefix = f"{base_name_transformer}.{layer_name}.{ind_layer}."
#         end = remove_prefix(k_orig, prefix)
#         new_k = f"{base_name_starter}.{layer_name}.{ind_layer_chunk}.{end}"
#         out_chunks["starter"][new_k] = model_params.pop(k_orig)

#     # ln_f - last layernorm
#     out_chunks["starter"][f"{base_name_starter}.ln_f.weight"] = model_params.pop(
#         f"{transformer_last}.weight"
#     )
#     if f"{transformer_last}.bias" in model_params.keys():
#         out_chunks["starter"][f"{base_name_starter}.ln_f.bias"] = model_params.pop(
#             f"{transformer_last}.bias"
#         )

#     # lm_head - final linear layers (not in 'transformer')
#     out_chunks["starter"][f"lm_head.weight"] = model_params.pop(
#         f"{output_layer}.weight"
#     )
#     if f"{output_layer}.bias" in model_params.keys():
#         out_chunks["starter"][f"lm_head.bias"] = model_params.pop(
#             f"{output_layer}.bias"
#         )

#     # 2. Select params for every Secondary
#     out_chunks["secondary"] = []
#     for i in range(1, n_nodes):
#         curr_params = {}

#         # Calculate valid layers indices in the original model
#         start_layer_ind = n_layers_start + (i - 1) * n_layers_secondary
#         finish_layer_ind = n_layers_start + i * n_layers_secondary
#         valid_layer_ind = list(range(start_layer_ind, finish_layer_ind))
#         relevant_keys = [
#             k
#             for k in list(model_params.keys())
#             if (
#                 k.startswith(f"{base_name_transformer}.{layer_name}")
#                 and int(k.split(".")[2]) in valid_layer_ind
#             )
#         ]

#         for k_orig in relevant_keys:
#             ind_layer = int(k_orig.split(".")[2])
#             ind_layer_chunk = ind_layer - start_layer_ind

#             prefix = f"{base_name_transformer}.{layer_name}.{ind_layer}."
#             end = remove_prefix(k_orig, prefix)
#             new_k = f"{base_name_secondary}.{layer_name}.{ind_layer_chunk}.{end}"
#             curr_params[new_k] = model_params.pop(k_orig)

#         out_chunks["secondary"].append(curr_params)

#     return out_chunks, layers_info


# def split_and_store(
#     model_params: Dict[str, Any],
#     n_nodes: int,
#     ckpt_dir: Union[Path, str],
#     **kwargs,
# ) -> Path:
#     """
#     Given a state dict, split it among a number of nodes following the configuration.

#     Args:
#         model_params: state dict
#         n_nodes: number of nodes among which to split the model
#         ckpt_dir: checkpoint directory of the model

#     Returns:
#         path of the chunks subdirectory (ckpt_dir/chunks/<n>nodes/)
#     """
#     if isinstance(ckpt_dir, str):
#         ckpt_dir = Path(ckpt_dir)

#     verb = False if "verb" not in kwargs else kwargs["verb"]

#     chunks, layer_info = split_parameters(model_params, n_nodes)
#     if len(model_params):
#         warnings.warn(f"{len(model_params)} elements have not been used")
#     del model_params
#     gc.collect()

#     n_secondary = n_nodes - 1

#     if verb:
#         print("Using the following split:")
#         print(f"- Starter node: {layer_info['N_LAYERS_START']} layers")
#         print(
#             f"- {n_secondary} secondary node{'s' if n_secondary - 1 else ''}: "
#             f"{layer_info['N_LAYERS_SECONDARY']} layers"
#         )

#     chunks_subfolder = ckpt_dir / "chunks" / f"{n_nodes}nodes"
#     os.makedirs(chunks_subfolder, exist_ok=True)

#     # Starter
#     starter_file = chunks_subfolder / "model_starter.pth"
#     torch.save(chunks["starter"], starter_file)

#     # Secondary (NOTE: zero-indexing in file name)
#     for i in range(n_secondary):
#         current_file = chunks_subfolder / f"model_secondary{i}.pth"
#         torch.save(chunks["secondary"][i], current_file)

#     return chunks_subfolder


def split_parameters_with_finisher(
    model_params: Dict[str, Any], n_nodes: int
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Split the model parameters with the final layer norm and output head on the last node.
    
    The parameters are divided as such:
        - Starter: token embedding + N_LAYERS_START transformer layers
        - Secondary (middle): N_LAYERS_SECONDARY transformer layers
        - Secondary (last/finisher): N_LAYERS_SECONDARY transformer layers + final layer norm + output head
    
    Args:
        model_params: complete model parameters (state dict)
        n_nodes: number of nodes among which to divide the parameters; must be >= 2
    
    Returns:
        dict containing:
            "starter": dict with the starter state dict
            "secondary": list containing the intermediate state dicts (last one is finisher)
        Layers information (n. layers per node)
    """
    assert n_nodes >= 2, "There must be at least 2 nodes in the network"
    
    # Set up parameter names
    base_name_transformer = "transformer"
    base_name_starter = "transformer"
    base_name_secondary = "transformer"
    tok_emb = "wte"
    layer_name = "h"
    transformer_last = f"{base_name_transformer}.ln_f"
    output_layer = "lm_head"
    
    # Count transformer layers
    layer_keys = [
        k
        for k in model_params.keys()
        if k.startswith(f"{base_name_transformer}.{layer_name}")
    ]
    layers_unique = list(set([".".join(k.split(".")[:3]) for k in layer_keys]))
    n_layers_model = len(layers_unique)
    if VERB:
        print(f"Number of transformer layers found in the model: {n_layers_model}")
    
    layers_info = {}
    n_layers_start = N_LAYERS_NODES[n_nodes][n_layers_model]["N_LAYERS_START"]
    layers_info["N_LAYERS_START"] = n_layers_start
    n_layers_secondary = N_LAYERS_NODES[n_nodes][n_layers_model]["N_LAYERS_SECONDARY"]
    layers_info["N_LAYERS_SECONDARY"] = n_layers_secondary
    
    if VERB:
        print(f"Number of layers - starter node: {n_layers_start}")
        print(f"Number of layers - secondary nodes: {n_layers_secondary}")
    
    out_chunks = {}
    
    # 1. Starter: embeddings + first N layers (NO ln_f, NO lm_head)
    out_chunks["starter"] = {}
    out_chunks["starter"][f"{base_name_starter}.{tok_emb}.weight"] = model_params.pop(
        f"{base_name_transformer}.{tok_emb}.weight"
    )
    if f"{base_name_transformer}.{tok_emb}.bias" in model_params.keys():
        out_chunks["starter"][f"{base_name_starter}.{tok_emb}.bias"] = model_params.pop(
            f"{base_name_transformer}.{tok_emb}.bias"
        )
    
    # Starter transformer layers
    valid_layer_ind = list(range(0, n_layers_start))
    relevant_keys = [
        k
        for k in list(model_params.keys())
        if (
            k.startswith(f"{base_name_transformer}.{layer_name}")
            and int(k.split(".")[2]) in valid_layer_ind
        )
    ]
    
    for k_orig in relevant_keys:
        ind_layer = int(k_orig.split(".")[2])
        ind_layer_chunk = ind_layer
        
        prefix = f"{base_name_transformer}.{layer_name}.{ind_layer}."
        end = k_orig[len(prefix):]
        new_k = f"{base_name_starter}.{layer_name}.{ind_layer_chunk}.{end}"
        out_chunks["starter"][new_k] = model_params.pop(k_orig)
    
    # 2. Secondary nodes: middle layers
    out_chunks["secondary"] = []
    for i in range(1, n_nodes):
        curr_params = {}
        
        # Calculate valid layer indices
        start_layer_ind = n_layers_start + (i - 1) * n_layers_secondary
        finish_layer_ind = n_layers_start + i * n_layers_secondary
        valid_layer_ind = list(range(start_layer_ind, finish_layer_ind))
        relevant_keys = [
            k
            for k in list(model_params.keys())
            if (
                k.startswith(f"{base_name_transformer}.{layer_name}")
                and int(k.split(".")[2]) in valid_layer_ind
            )
        ]
        
        for k_orig in relevant_keys:
            ind_layer = int(k_orig.split(".")[2])
            ind_layer_chunk = ind_layer - start_layer_ind
            
            prefix = f"{base_name_transformer}.{layer_name}.{ind_layer}."
            end = k_orig[len(prefix):]
            new_k = f"{base_name_secondary}.{layer_name}.{ind_layer_chunk}.{end}"
            curr_params[new_k] = model_params.pop(k_orig)
        
        # 3. If this is the LAST secondary node, add ln_f and lm_head
        if i == n_nodes - 1:
            # Final layer norm
            curr_params[f"{base_name_secondary}.ln_f.weight"] = model_params.pop(
                f"{transformer_last}.weight"
            )
            if f"{transformer_last}.bias" in model_params.keys():
                curr_params[f"{base_name_secondary}.ln_f.bias"] = model_params.pop(
                    f"{transformer_last}.bias"
                )
            
            # Output head
            curr_params["lm_head.weight"] = model_params.pop(f"{output_layer}.weight")
            if f"{output_layer}.bias" in model_params.keys():
                curr_params["lm_head.bias"] = model_params.pop(f"{output_layer}.bias")
        
        out_chunks["secondary"].append(curr_params)
    
    return out_chunks, layers_info


def split_and_store_with_finisher(
    model_params: Dict[str, Any],
    n_nodes: int,
    ckpt_dir: Union[Path, str],
    **kwargs,
) -> Path:
    """
    Split and store model with final layers on the last secondary node (finisher pattern).
    
    Args:
        model_params: state dict
        n_nodes: number of nodes among which to split the model
        ckpt_dir: checkpoint directory of the model
    
    Returns:
        path of the chunks subdirectory (ckpt_dir/chunks/<n>nodes_finisher/)
    """
    if isinstance(ckpt_dir, str):
        ckpt_dir = Path(ckpt_dir)
    
    verb = False if "verb" not in kwargs else kwargs["verb"]
    
    chunks, layer_info = split_parameters_with_finisher(model_params, n_nodes)
    if len(model_params):
        warnings.warn(f"{len(model_params)} elements have not been used")
    del model_params
    gc.collect()
    
    n_secondary = n_nodes - 1
    
    if verb:
        print("Using the following split (with finisher):")
        print(f"- Starter node: {layer_info['N_LAYERS_START']} layers")
        
        # Middle secondary nodes (if any)
        n_middle = n_secondary - 1
        if n_middle > 0:
            print(
                f"- {n_middle} middle secondary node{'s' if n_middle > 1 else ''}: "
                f"{layer_info['N_LAYERS_SECONDARY']} layers each"
            )
        
        # Finisher node (always present)
        print(
            f"- 1 finisher node: {layer_info['N_LAYERS_SECONDARY']} layers + "
            "final norm + output head"
        )
    
    chunks_subfolder = ckpt_dir / "chunks" / f"{n_nodes}nodes_finisher"
    os.makedirs(chunks_subfolder, exist_ok=True)
    
    # Starter
    starter_file = chunks_subfolder / "model_starter.pth"
    torch.save(chunks["starter"], starter_file)
    
    # Secondary nodes (last one is finisher)
    for i in range(n_secondary):
        current_file = chunks_subfolder / f"model_secondary{i}.pth"
        torch.save(chunks["secondary"][i], current_file)
    
    return chunks_subfolder


def serialize_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Serialize a mapping, specifically a state dict, to allow it to be read as a
    JSON/dict.
    """
    json_serializable_params = {}
    for key, value in params.items():
        json_serializable_params[key] = (
            value.tolist() if isinstance(value, torch.Tensor) else value
        )

    return json_serializable_params


def deserialize_params(params: Dict) -> Mapping[str, Any]:
    """
    De-serialize a dictionary and return a state dict containing torch model parameters.
    """
    deserialized_params = {}
    for key, value in params.items():
        if isinstance(value, list):
            # Convert lists back to PyTorch tensors
            deserialized_params[key] = torch.tensor(value)
        else:
            deserialized_params[key] = value

    return deserialized_params


def count_transformer_blocks(
    state_dict: Dict[str, Any], base_name_transformer: Optional[str] = "transformer"
) -> int:
    """
    Given a state dict, return the number of detected transformer blocks.
    The default name for the transformer blocks is `transformer`, but can be overridden
    to support a different naming convention.

    Args:
        state_dict: dict containing the model parameters.
        base_name_transformer: base name of the transformer block, i.e., first "key" in
            the dict.
    """
    layer_name = "h"

    # Count the number of detected transformer layers
    layer_keys = [
        k
        for k in state_dict.keys()
        if k.startswith(f"{base_name_transformer}.{layer_name}")
    ]
    layers_unique = list(set([".".join(k.split(".")[:3]) for k in layer_keys]))
    return len(layers_unique)


def load_sd( 
    model_path: Path, device: Optional[Union[torch.device, str]] = "cpu", **kwargs
) -> Dict[str, Any]:
    """
    Load a state dictionary (model parameters).

    Args:
        model_path: path of the file (typically .pt or .pth) containing the model
            parameters.
        device (default "cpu"): device where to load the weights (NOT the model!)

    Returns:
        state dict of the model (can be passed to a compatible nn.Module object through
            the method `nn.Module.load_state_dict()`.
    """
    # PyTorch 2.6+ defaults to weights_only=True for security
    # We need to set it to False to load LLM model files
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    
    try:
        sd = torch.load(model_path, map_location=device, **kwargs)
    except Exception as e:
        if "out of memory" in str(e):
            if device != "cpu":
                warnings.warn(
                    f"Unable to fit model ckpt in {device} memory! Retrying with cpu"
                )
                sd = torch.load(model_path, map_location="cpu", weights_only=False)
            else:
                raise e
        else:
            raise e

    return sd


def load_from_pt(
    model_path: Union[Path, str],
    device: Optional[Union[torch.device, str]] = "cpu",
    config_only: Optional[bool] = False,
) -> Tuple[Config, Optional[Dict[str, Any]]]:
    """
    Load model weights from disk.

    Args:
        model_path: path to the checkpoint
        device (default: "cpu"): device where to load state dict; default: "cpu"
        config_only (default: False): if True, only return the Config object

    Returns:
        model config (Config object)
        [model state dictionary, compatible with GPT class]
    """
    if isinstance(model_path, str):
        model_dir = Path(model_path)
    elif isinstance(model_path, Path):
        model_dir = model_path
    else:
        raise TypeError

    if not model_dir.is_dir():
        raise NotADirectoryError(f"Unable to find model checkpoint at {model_dir}")

    config = Config.from_file(model_dir / "model_config.yaml")

    if config_only:
        return config, None

    pth_file = model_dir / "lit_model.pth"
    sd = load_sd(pth_file, device)

    return config, sd


def load_from_hf(
    repo_id: str,
    access_token: Optional[str] = None,
    dtype: Optional[str] = None,
    checkpoint_dir: Path = Path("checkpoints"),
    model_name: Optional[str] = None,
    device: Optional[str] = "cpu",
    config_only: Optional[bool] = False,
    convert_checkpoint: Optional[bool] = True,
) -> Tuple[Config, Optional[Dict[str, Any]]]:
    """
    Load model weights from Huggingface.
    It saves the files to the checkpoint directory, converts them to the right format
    and loads the model configuration and the state dict.

    Args:
        repo_id: Huggingface Hub repository ID
        access_token: optional API token for accessing private Huggingface models
        dtype: data type for the downloaded weights
        checkpoint_dir: path of the directory where to place the model folders
        model_name: the existing config name to use for this `repo_id`. This is
            useful to download alternative weights of existing architectures.
        device: device where to load state dict
        config_only (default: False): if true, only return the Config object (note that
            the model will be downloaded anyways)
        convert_checkpoint (default: True): if False, skip HF to LitGPT conversion

    Returns:
        model config (Config object)
        [model state dictionary, compatible with GPT class]
    """
    from .download import download_from_hub

    download_from_hub(
        repo_id=repo_id,
        access_token=access_token,
        dtype=dtype,
        checkpoint_dir=checkpoint_dir,
        model_name=model_name,
        convert_checkpoint=convert_checkpoint,
    )

    model_path = checkpoint_dir / repo_id
    return load_from_pt(model_path, device, config_only=config_only)


def load_from_hf_direct(
    model_path: Union[Path, str],
    device: Optional[Union[torch.device, str]] = "cpu",
    dtype: Optional[torch.dtype] = None,
    config_only: Optional[bool] = False,
) -> Tuple[Config, Optional[Dict[str, Any]]]:
    """
    Load model weights directly from HF format (safetensors/bin) without converting to LitGPT format on disk.
    The conversion happens in memory only.

    Args:
        model_path: path to the HF checkpoint directory
        device (default: "cpu"): device where to load state dict
        dtype: optional dtype to convert weights to
        config_only (default: False): if True, only return the Config object

    Returns:
        model config (Config object)
        [model state dictionary, compatible with GPT class]
    """
    import json
    import gc
    from functools import partial
    from .convert_hf_checkpoint import copy_weights_hf_llama, copy_weights_gpt_neox, copy_weights_falcon, copy_weights_phi
    
    if isinstance(model_path, str):
        model_dir = Path(model_path)
    elif isinstance(model_path, Path):
        model_dir = model_path
    else:
        raise TypeError(f"model_path must be str or Path, got {type(model_path)}")

    if not model_dir.is_dir():
        raise NotADirectoryError(f"Unable to find model checkpoint at {model_dir}")

    # Load config from HF config.json
    hf_config_path = model_dir / "config.json"
    if not hf_config_path.exists():
        raise FileNotFoundError(f"HF config.json not found at {hf_config_path}")
    
    with open(hf_config_path, "r", encoding="utf-8") as f:
        hf_config = json.load(f)
    
    # Try to get the model name from config or directory structure
    model_name = hf_config.get("_name_or_path", "") or hf_config.get("model_type", "")
    if not model_name:
        # Try to infer from directory name
        model_name = model_dir.name.lower()
    
    # Create Config from HF config
    config = Config.from_hf_config(hf_config, model_name=model_name)
    
    # Save the config for future use (so load_from_pt can work)
    save_config(config, model_dir)
    
    if config_only:
        return config, None
    
    # Determine copy function based on model architecture
    if config.mlp_class_name in ("LLaMAMLP", "GemmaMLP", "LLaMAMoE"):
        qkv_weights = {}
        copy_fn = partial(copy_weights_hf_llama, config, qkv_weights)
    elif "falcon" in model_name.lower():
        copy_fn = partial(copy_weights_falcon, model_name)
    elif "phi" in model_name.lower():
        qkv_weights = {}
        copy_fn = partial(copy_weights_phi, config, qkv_weights)
    else:
        copy_fn = copy_weights_gpt_neox
    
    # Find weight files
    safetensor_files = list(model_dir.glob("*.safetensors"))
    bin_files = list(model_dir.glob("*.bin"))
    # Filter out training_args.bin
    bin_files = [f for f in bin_files if f.name != "training_args.bin"]
    
    sd = {}
    
    if safetensor_files:
        # Load from safetensors directly
        try:
            from safetensors.torch import load_file as safetensors_load
        except ImportError:
            raise ImportError("safetensors package is required to load .safetensors files")
        
        for sf_file in sorted(safetensor_files):
            print(f"Loading {sf_file}")
            hf_weights = safetensors_load(sf_file, device=str(device))
            copy_fn(sd, hf_weights, saver=None, dtype=dtype)
            del hf_weights
            gc.collect()
    elif bin_files:
        # Load from bin files
        for bin_file in sorted(bin_files):
            print(f"Loading {bin_file}")
            hf_weights = torch.load(bin_file, map_location=device)
            copy_fn(sd, hf_weights, saver=None, dtype=dtype)
            del hf_weights
            gc.collect()
    else:
        raise ValueError(f"No weight files (.safetensors or .bin) found in {model_dir}")
    
    # Move state dict to device if needed
    if device != "cpu":
        for key in sd:
            if isinstance(sd[key], torch.Tensor):
                sd[key] = sd[key].to(device)
    
    gc.collect()
    
    return config, sd


def save_config(config: "Config", checkpoint_dir: Path) -> None:
    config_path = checkpoint_dir / "model_config.yaml"
    # Skip if file already exists (don't overwrite manually configured files)
    if config_path.exists():
        return
    config_dict = asdict(config)
    with open(config_path, "w", encoding="utf-8") as fp:
        yaml.dump(config_dict, fp)


def init_from_state_dict(model: nn.Module, state_dict: Dict[str, Any]) -> nn.Module:
    """
    This method is used to fill up a model (`torch.nn.Module`) that is currently loaded
    as "meta" (i.e., empty) with the parameters and buffers stored in `state_dict`.

    Args:
        model: empty model (stored in "meta")
        state_dict
        device (optional): if missing will be the same as the device of state_dict
    """
    if not next(model.parameters()).device == torch.device("meta"):
        raise RuntimeError("The model is not on 'meta' - it is using memory space")

    keys_to_submodule = get_keys_to_submodule(model)
    for key, submodule in keys_to_submodule.items():
        # get the value from the state_dict
        val = state_dict[key]
        # The key is composed of <name>.<subname>.<subsubname>
        # The actual submodule's parameter is stored inside the last subname.
        # E.g., if key is `in_proj.weight`, the correct field if `weight`
        param_name = key.split('.')[-1]
        # Keep the dtype of the state dict
        val = val
        # Create a new parameter
        new_val = torch.nn.Parameter(val, requires_grad=False)
        setattr(submodule, param_name, new_val)

    return model


def get_keys_to_submodule(model: nn.Module) -> Dict[str, nn.Module]:
    keys_to_submodule = {}
    # iterate all submodules
    for submodule_name, submodule in model.named_modules():
        # iterate all paramters in each submobule
        for param_name, param in submodule.named_parameters():
            # param_name is organized as <name>.<subname>.<subsubname> ...
            # the more we go deep in the model, the less "subname"s we have
            splitted_param_name = param_name.split('.')
            # if we have only one subname, then it means that we reach a "leaf" submodule, 
            # we cannot go inside it anymore. This is the actual parameter
            is_leaf_param = len(splitted_param_name) == 1
            if is_leaf_param:
                # we recreate the correct key
                key = f"{submodule_name}.{param_name}"
                # we associate this key with this submodule
                keys_to_submodule[key] = submodule
                
    return keys_to_submodule
