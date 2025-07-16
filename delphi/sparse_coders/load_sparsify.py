from functools import partial
from pathlib import Path
from typing import Callable

import torch
from sparsify import SparseCoder
from torch import Tensor
from transformers import PreTrainedModel


def sae_dense_latents(x: Tensor, sae: SparseCoder) -> Tensor:
    """Run `sae` on `x`, yielding the dense activations."""
    x_in = x.reshape(-1, x.shape[-1]) # si lavora sempre in batch
    encoded = sae.encode(x_in)
    buf = torch.zeros(
        x_in.shape[0], sae.num_latents, dtype=x_in.dtype, device=x_in.device
    )

    # inserisce le attivazioni sparse nel buffer denso (le posizioni non attivate restano a zero)
    buf = buf.scatter_(-1, encoded.top_indices, encoded.top_acts.to(buf.dtype))
    

    """
    Infine si ripristina la forma originale del batch: 
    se all'inizio x.shape = (32, 10, 768) batch di 32 sequenze, ognuna lunga 10 token, con 768 dimensioni 
    e sae.num_latents = 4096
    con il seguente reshape passo da buf.shape = (320, 4096) a buf.shape = (32, 10, 4096).
    """
    return buf.reshape(*x.shape[:-1], -1)


def resolve_path(
    model: PreTrainedModel | torch.nn.Module, path_segments: list[str]
) -> list[str] | None:
    """Attempt to resolve the path segments to the model in the case where it
    has been wrapped (e.g. by a LanguageModel, causal model, or classifier)."""
    # If the first segment is a valid attribute, return the path segments
    if hasattr(model, path_segments[0]):
        return path_segments

    # Look for the first actual model inside potential wrappers
    for attr_name, attr in model.named_children():
        if isinstance(attr, torch.nn.Module):
            print(f"Checking wrapper model attribute: {attr_name}")
            if hasattr(attr, path_segments[0]):
                print(
                    f"Found matching path in wrapper at {attr_name}.{path_segments[0]}"
                )
                return [attr_name] + path_segments

            # Recursively check deeper
            deeper_path = resolve_path(attr, path_segments)
            if deeper_path is not None:
                print(f"Found deeper matching path starting with {attr_name}")
                return [attr_name] + deeper_path
    return None


def load_sparsify_sparse_coders(
    name: str,
    hookpoints: list[str],
    device: str | torch.device,
    compile: bool = False,
) -> dict[str, SparseCoder]:
    """
    Load sparsify sparse coders for specified hookpoints.

    Args:
        model (Any): The model to load autoencoders for.
        name (str): The name of the sparse model to load. If the model is on-disk
            this is the path to the directory containing the sparse model weights.
        hookpoints (list[str]): list of hookpoints to identify the sparse models.
        device (str | torch.device | None, optional): The device to load the
            sparse models on. If not specified the sparse models will be loaded
            on the same device as the base model.

    Returns:
        dict[str, Any]: A dictionary mapping hookpoints to sparse models.
    """

    # Load the sparse models from local path
    sparse_model_dict = {}
    name_path = Path(name)
    if name_path.exists(): # è stato specificato un solo hookpoint
        for hookpoint in hookpoints:
            sparse_model_dict[hookpoint] = SparseCoder.load_from_disk(
                name_path / hookpoint, device=device
            )
            if compile:
                sparse_model_dict[hookpoint] = torch.compile(
                    sparse_model_dict[hookpoint]
                )
    else: # sono stati specificati più hookpoint
        # Load on CPU first to not run out of memory
        sparse_models = SparseCoder.load_many(name, device="cpu")
        for hookpoint in hookpoints:
            sparse_model_dict[hookpoint] = sparse_models[hookpoint].to(device)
            if compile:
                sparse_model_dict[hookpoint] = torch.compile(
                    sparse_model_dict[hookpoint]
                )

        del sparse_models
    return sparse_model_dict


def load_sparsify_hooks(
    model: PreTrainedModel,
    name: str,
    hookpoints: list[str],
    device: str | torch.device | None = None,
    compile: bool = False,
) -> tuple[dict[str, Callable], bool]:
    """
    Load the encode functions for sparsify sparse coders on specified hookpoints.

    Args:
        model (Any): The model to load autoencoders for.
        name (str): The name of the sparse model to load. If the model is on-disk
            this is the path to the directory containing the sparse model weights.
        hookpoints (list[str]): list of hookpoints to identify the sparse models.
        device (str | torch.device | None, optional): The device to load the
            sparse models on. If not specified the sparse models will be loaded
            on the same device as the base model.

    Returns:
        dict[str, Callable]: A dictionary mapping hookpoints to encode functions.
    """
    device = model.device or "cpu"

    # Caricamento degli autoencoder sparsi (da path locale oppure scaricandoli da Hugging Face)
    sparse_model_dict = load_sparsify_sparse_coders(
        name,
        hookpoints,
        device,
        compile,
    )
    hookpoint_to_sparse_encode = {}
    transcode = False
    for hookpoint, sparse_model in sparse_model_dict.items():
        print(f"Resolving path for hookpoint: {hookpoint}")
        path_segments = resolve_path(model, hookpoint.split("."))
        if path_segments is None:
            raise ValueError(f"Could not find valid path for hookpoint: {hookpoint}")

        hookpoint_to_sparse_encode[".".join(path_segments)] = partial(
            sae_dense_latents, sae=sparse_model
        )
        # We only need to check if one of the sparse models is a transcoder
        if hasattr(sparse_model.cfg, "transcode"):
            if sparse_model.cfg.transcode:
                transcode = True
        if hasattr(sparse_model.cfg, "skip_connection"):
            if sparse_model.cfg.skip_connection:
                transcode = True

    """
    Per esempio, se specifichi hookpoints=['layers.5']:
    Viene caricato un autoencoder specifico per il layer 5
    Viene creata una funzione che intercetta le attivazioni in quel punto e le passa attraverso l'autoencoder
    """

    return hookpoint_to_sparse_encode, transcode
    
