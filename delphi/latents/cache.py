import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from jaxtyping import Float, Int
from safetensors.numpy import save_file
from torch import Tensor
from tqdm import tqdm
from transformers import PreTrainedModel

from delphi.config import CacheConfig
from delphi.latents.collect_activations import collect_activations

import sys

location_tensor_shape = Float[Tensor, "batch sequence num_latents"]
token_tensor_shape = Float[Tensor, "batch sequence"]


class InMemoryCache:
    """
    The Cache class stores latent locations and activations for modules.
    It provides methods for adding, saving, and retrieving non-zero activations.
    """

    def __init__(
        self,
        filters: dict[str, Float[Tensor, "indices"]] | None = None,
        batch_size: int = 64,
    ):
        """
        Initialize the Cache.

        Args:
            filters: Filters for selecting specific latents.
            batch_size: Size of batches for processing. Defaults to 64.
        """
        self.latent_locations_batches: dict[str, list[location_tensor_shape]] = (
            defaultdict(list)
        )
        self.latent_activations_batches: dict[str, list[location_tensor_shape]] = (
            defaultdict(list)
        )
        self.tokens_batches: dict[str, list[token_tensor_shape]] = defaultdict(list)

        self.latent_locations: dict[str, location_tensor_shape] = {}
        self.latent_activations: dict[str, location_tensor_shape] = {}
        self.tokens: dict[str, token_tensor_shape] = {}

        self.filters = filters
        self.batch_size = batch_size

    def add(
        self,
        latents: location_tensor_shape,
        tokens: token_tensor_shape,
        batch_number: int,
        module_path: str,
    ):
        """
        Add the latents from a module to the cache.

        Args:
            latents: Latent activations.
            tokens: Input tokens.
            batch_number: Current batch number.
            module_path: Path of the module.
        """
        latent_locations, latent_activations = self.get_nonzeros(latents, module_path)
        latent_locations = latent_locations.cpu()
        latent_activations = latent_activations.cpu()
        tokens = tokens.cpu()

        # Adjust batch indices
        latent_locations[:, 0] += batch_number * self.batch_size
        self.latent_locations_batches[module_path].append(latent_locations)
        self.latent_activations_batches[module_path].append(latent_activations)
        self.tokens_batches[module_path].append(tokens)

    def save(self):
        """
        Concatenate the latent locations and activations for all modules.
        """
        for module_path in self.latent_locations_batches.keys():
            self.latent_locations[module_path] = torch.cat(
                self.latent_locations_batches[module_path], dim=0
            )

            self.latent_activations[module_path] = torch.cat(
                self.latent_activations_batches[module_path], dim=0
            )

            self.tokens[module_path] = torch.cat(
                self.tokens_batches[module_path], dim=0
            )

    def get_nonzeros_batch(
        self, latents: location_tensor_shape
    ) -> tuple[
        Float[Tensor, "batch sequence num_latents"], Float[Tensor, "batch sequence "]
    ]:
        """
        Get non-zero activations for large batches that exceed int32 max value.

        Args:
            latents: Input latent activations.

        Returns:
            tuple[Tensor, Tensor]: Non-zero latent locations and activations.
        """
        # Calculate the maximum batch size that fits within sys.maxsize
        max_batch_size = torch.iinfo(torch.int32).max // (
            latents.shape[1] * latents.shape[2]
        )
        nonzero_latent_locations = []
        nonzero_latent_activations = []

        for i in range(0, latents.shape[0], max_batch_size):
            batch = latents[i : i + max_batch_size]

            # Get nonzero locations and activations
            batch_locations = torch.nonzero(batch.abs() > 1e-5)
            batch_activations = batch[batch.abs() > 1e-5]

            # Adjust indices to account for batching
            batch_locations[:, 0] += i
            nonzero_latent_locations.append(batch_locations)
            nonzero_latent_activations.append(batch_activations)

        # Concatenate results
        nonzero_latent_locations = torch.cat(nonzero_latent_locations, dim=0)
        nonzero_latent_activations = torch.cat(nonzero_latent_activations, dim=0)
        return nonzero_latent_locations, nonzero_latent_activations

    def get_nonzeros(self, latents: location_tensor_shape, module_path: str) -> tuple[
        location_tensor_shape,
        location_tensor_shape,
    ]:
        """
        Get the nonzero latent locations and activations.

        Args:
            latents: Input latent activations.
            module_path: Path of the module.

        Returns:
            tuple[Tensor, Tensor]: Non-zero latent locations and activations.
        """
        size = latents.shape[1] * latents.shape[0] * latents.shape[2]
        if size > torch.iinfo(torch.int32).max:
            (
                nonzero_latent_locations,
                nonzero_latent_activations,
            ) = self.get_nonzeros_batch(latents)
        else:
            nonzero_latent_locations = torch.nonzero(latents.abs() > 1e-5)
            nonzero_latent_activations = latents[latents.abs() > 1e-5]

        # Return all nonzero latents if no filter is provided
        if self.filters is None:
            return nonzero_latent_locations, nonzero_latent_activations

        # Return only the selected latents if a filter is provided
        else:
            selected_latents = self.filters[module_path]
            mask = torch.isin(nonzero_latent_locations[:, 2], selected_latents)

            return nonzero_latent_locations[mask], nonzero_latent_activations[mask]


class LatentCache:
    """
    LatentCache manages the caching of latent activations for a model.
    Handles the process of running the model, storing activations, and saving to disk
    """

    def __init__(
        self,
        model: PreTrainedModel,
        hookpoint_to_sparse_encode: dict[str, Callable],
        batch_size: int,
        transcode: bool = False,
        filters: dict[str, Float[Tensor, "indices"]] | None = None,
        log_path: Path | None = None,
    ):
        """
        Initialize the LatentCache.

        Args:
            model: The model to cache latents for.
            hookpoint_to_sparse_encode: Dictionary of sparse encoding functions.
            batch_size: Size of batches for processing.
            transcode: Whether to transcode the model outputs.
            filters: Filters for selecting specific latents.
            log_path: Path to save logging output.
        """
        self.model = model
        self.hookpoint_to_sparse_encode = hookpoint_to_sparse_encode
        self.transcode = transcode
        self.batch_size = batch_size
        self.width = None
        self.cache = InMemoryCache(filters, batch_size=batch_size)
        self.hookpoint_firing_counts: dict[str, Tensor] = {}

        self.log_path = log_path
        if filters is not None:
            self.filter_submodules(filters)

    def load_token_batches(
        self, n_tokens: int, tokens: token_tensor_shape
    ) -> list[token_tensor_shape]:
        """
        Load and prepare token batches for processing.

        Args:
            n_tokens: Total number of tokens to process.
            tokens: Input tokens.

        Returns:
            list[Tensor]: list of token batches.
        """
        print(f"Numero totale di token da processare: {n_tokens}")
        print(f"Forma dei token in input: {tokens.shape}")

        max_batches = n_tokens // tokens.shape[1] # tokens.shape[1] sarebbe pari a cache_ctx_len
        print(f"Numero massimo di batch calcolato: {max_batches}")

        tokens = tokens[:max_batches]
        print(f"Token dopo il taglio: {tokens.shape}")

        n_mini_batches = len(tokens) // self.batch_size
        print(f"Numero di mini-batch: {n_mini_batches}")

        token_batches = [
            tokens[self.batch_size * i : self.batch_size * (i + 1), :]
            for i in range(n_mini_batches)
        ] # itero sulle righe di "tokens" ed estraggo i batch

        print(f"Numero di batch creati: {len(token_batches)}")
        if len(token_batches) > 0:
            print(f"Forma del primo batch: {token_batches[0].shape}")

        return token_batches

    def filter_submodules(self, filters: dict[str, Float[Tensor, "indices"]]):
        """
        Filter submodules based on the provided filters.

        Args:
            filters: Filters for selecting specific latents.
        """
        filtered_submodules = {}
        for hookpoint in self.hookpoint_to_sparse_encode.keys():
            if hookpoint in filters:
                filtered_submodules[hookpoint] = self.hookpoint_to_sparse_encode[
                    hookpoint
                ]
        self.hookpoint_to_sparse_encode = filtered_submodules

    def run(self, n_tokens: int, tokens: token_tensor_shape):
        """
        Run the latent caching process.

        Args:
            n_tokens: Total number of tokens to process.
            tokens: Input tokens.
        """
        
        token_batches = self.load_token_batches(n_tokens, tokens)
        print("Numero di batch creati:", len(token_batches))
        print("Numero di token nel primo batch:", token_batches[0].numel()) 

        total_tokens = 0
        total_batches = len(token_batches)
        tokens_per_batch = token_batches[0].numel()
        with tqdm(total=total_batches, desc="Caching latents") as pbar:
            for batch_number, batch in enumerate(token_batches):
                total_tokens += tokens_per_batch

                with torch.no_grad():
                    with collect_activations(self.model, list(self.hookpoint_to_sparse_encode.keys()), self.transcode, ) as activations:
                        
                        # 1. Passa i token attraverso il modello base
                        self.model(batch.to(self.model.device))

                        # 2. Per ogni hookpoint (A specifici "hookpoints" nel modello, le attivazioni vengono intercettate)
                        for hookpoint, latents in activations.items():

                            # 3. Passa le attivazioni attraverso l'autoencoder sparso corrispondente
                            sae_latents = self.hookpoint_to_sparse_encode[hookpoint](
                                latents
                            )

                            # 4. Aggiungi le attivazioni all'in-memory cache
                            self.cache.add(sae_latents, batch, batch_number, hookpoint)

                            # Viene tenuto traccia di quali neuroni si "attivano" più frequentemente
                            firing_counts = (sae_latents > 0).sum((0, 1))

                            if self.width is None:
                                self.width = sae_latents.shape[2]

                            if hookpoint not in self.hookpoint_firing_counts:
                                self.hookpoint_firing_counts[hookpoint] = (
                                    firing_counts.cpu()
                                )
                            else:
                                self.hookpoint_firing_counts[
                                    hookpoint
                                ] += firing_counts.cpu()

                # Update the progress bar
                pbar.update(1)
                pbar.set_postfix({"Total Tokens": f"{total_tokens:,}"})

        print(f"Total tokens processed: {total_tokens:,}")
        self.cache.save()
        self.save_firing_counts()

    def save(self, save_dir: Path, save_tokens: bool = True):
        """
        Save the cached latents to disk.

        Args:
            save_dir: Directory to save the latents.
            save_tokens: Whether to save the dataset tokens used to generate the cache.
            Defaults to True.
        """
        for module_path in self.cache.latent_locations.keys():
            output_file = save_dir / f"{module_path}.safetensors"

            data = {
                "locations": self.cache.latent_locations[module_path].numpy(),
                "activations": self.cache.latent_activations[module_path].numpy(),
            }
            if save_tokens:
                data["tokens"] = self.cache.tokens[module_path].numpy()

            save_file(data, output_file)

    def _generate_split_indices(self, n_splits: int) -> list[tuple[Tensor, Tensor]]:
        """
        Generate indices for splitting the latent space.

        Args:
            n_splits: Number of splits to generate.

        Returns:
            list[tuple[int, int]]: list of start and end indices for each split.
        """
        assert self.width is not None, "Width must be set before generating splits"
        boundaries = torch.linspace(0, self.width, steps=n_splits + 1).long()

        # Adjust end by one
        return list(zip(boundaries[:-1], boundaries[1:] - 1))

    def save_splits(self, n_splits: int, save_dir: Path, save_tokens: bool = True):
        """
        Save the cached non-zero latent activations and locations in splits.

        Args:
            n_splits: Number of splits to generate.
            save_dir: Directory to save the splits.
            save_tokens: Whether to save the dataset tokens used to generate the cache.
            Defaults to True.
        """
        split_indices = self._generate_split_indices(n_splits)
        for module_path in self.cache.latent_locations.keys():
            latent_locations = self.cache.latent_locations[module_path]
            latent_activations = self.cache.latent_activations[module_path]
            tokens = self.cache.tokens[module_path].numpy()

            latent_indices = latent_locations[:, 2]

            for start, end in split_indices:
                mask = (latent_indices >= start) & (latent_indices <= end)

                masked_activations = latent_activations[mask].half().numpy()

                masked_locations = latent_locations[mask].numpy()

                # Optimization to reduce the max value to enable a smaller dtype
                masked_locations[:, 2] = masked_locations[:, 2] - start.item()

                if (
                    masked_locations[:, 2].max() < 2**16
                    and masked_locations[:, 0].max() < 2**16
                ):
                    masked_locations = masked_locations.astype(np.uint16)
                else:
                    masked_locations = masked_locations.astype(np.uint32)
                    print(
                        "Warning: Increasing the number of splits might reduce the"
                        "memory usage of the cache."
                    )

                module_dir = save_dir / module_path
                module_dir.mkdir(parents=True, exist_ok=True)

                output_file = module_dir / f"{start}_{end}.safetensors"

                split_data = {
                    "locations": masked_locations,
                    "activations": masked_activations,
                }
                if save_tokens:
                    split_data["tokens"] = tokens

                save_file(split_data, output_file)

    def generate_statistics_cache(self):
        """
        Print statistics (number of dead features, number of single token features)
        to the console.
        """
        
    """
    
        assert self.width is not None, "Width must be set before generating statistics"
        print("Feature statistics:")
        # Token frequency
        for module_path in self.cache.latent_locations.keys():
            print(f"# Module: {module_path}")
            generate_statistics_cache(
                self.cache.tokens[module_path],
                self.cache.latent_locations[module_path],
                self.cache.latent_activations[module_path],
                self.width,
                verbose=True,
            )
    """

    def save_config(self, save_dir: Path, cfg: CacheConfig, model_name: str):
        """
        Save the configuration for the cached latents.

        Args:
            save_dir: Directory to save the configuration.
            cfg: Configuration object.
            model_name: Name of the model.
        """
        for module_path in self.cache.latent_locations.keys():
            config_file = save_dir / module_path / "config.json"
            with open(config_file, "w") as f:
                config_dict = cfg.to_dict()
                config_dict["model_name"] = model_name
                json.dump(config_dict, f, indent=4)

    def save_firing_counts(self):
        """
        Save the firing counts for the cached latents.
        """
        if self.log_path is None:
            log_path = Path.cwd() / "results" / "log" / "hookpoint_firing_counts.pt"
        else:
            log_path = self.log_path / "hookpoint_firing_counts.pt"

        """
        il file hookpoint_firing_counts.pt contiene un dizionario che tiene traccia di quante volte ogni neurone 
        nell'autoencoder sparso si è "attivato" durante l'elaborazione dei token. Specificamente:
        Struttura del file:
        È un dizionario Python dove le chiavi sono i nomi degli hookpoint (es. "layers.5")
        Per ogni hookpoint, contiene un tensore PyTorch che registra il conteggio delle attivazioni per ogni neurone
        """

        log_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.hookpoint_firing_counts, log_path)


@dataclass
class CacheStatistics:
    frac_alive: float
    frac_fired_1pct: float
    frac_fired_10pct: float
    frac_weak_single_token: float
    frac_strong_single_token: float


@torch.inference_mode()
def generate_statistics_cache(
    tokens: Int[Tensor, "batch sequence"], # tensore che contiene batch di sequenze di token (ogni riga è una sequenza di token)
    latent_locations: Int[Tensor, "n_activations 3"], # per ogni attivazione latente ci sono 3 coordinate: [batch_idx, seq_idx, feature_idx]
    activations: Float[Tensor, "n_activations"], # per ciascuna attivazione latente viene memorizzato un valore di attivazione (l'intensità dell'attivazione)
    width: int,
    verbose: bool = False,
) -> CacheStatistics:
    """Generate global statistics for the cache."

    Args:
        tokens (Int[Tensor, "batch sequence"]): Tokens used to generate the cache.
        latent_locations (Int[Tensor, "n_activations 3"]): Indices of the latent
            activations, corresponding to `tokens`.
        activations (Float[Tensor, "n_activations"]): Activations of the latents,
            as stored by the cache.
        width (int): Width of the cache to test.
        verbose (bool, optional): Print results to stdout. Defaults to False.
    Returns:
        CacheStatistics: the statistics
    """
    total_n_tokens = tokens.shape[0] * tokens.shape[1]

    latent_locations, latents = latent_locations[:, :2], latent_locations[:, 2]

    # torch always sorts for unique, so we might as well do it
    sorted_latents, latent_indices = latents.sort()
    sorted_activations = activations[latent_indices]
    sorted_tokens = tokens[latent_locations[latent_indices]]

    unique_latents, counts = torch.unique_consecutive(
        sorted_latents, return_counts=True
    ) # counts: per ogni valore unico in unique_latents, restituisce il numero di volte consecutive in cui quel valore appare nel tensore ordinato

    # How many unique latents ever activated on the cached tokens
    num_alive = counts.shape[0]
    fraction_alive = num_alive / width # width è il numero di neuroni nell'autoencoder
    if verbose:
        print(f"Fraction of latents alive: {fraction_alive:%}") # percentuale di neuroni nell'autoencoder che si sono attivati almeno una volta durante l'elaborazione dei token
    # Compute densities of latents
    densities = counts / total_n_tokens # indica quanto frequentemente ogni latente viene attivato rispetto al totale dei token elaborati

    # How many fired more than 1% of the time
    one_percent = (densities > 0.01).sum() / width
    # How many fired more than 10% of the time
    ten_percent = (densities > 0.1).sum() / width
    if verbose:
        print(f"Fraction of latents fired more than 1% of the time: {one_percent:%}") # Frazione di latenti che si attivano più dell'1% delle volte (Utile per identificare i neuroni più attivi e stabili)
        print(f"Fraction of latents fired more than 10% of the time: {ten_percent:%}") # Frazione di latenti che si attivano più del 10% delle volte (Aiuta a identificare i neuroni più specializzati e frequentemente utilizzati)
    
    # Try to estimate simple feature frequency
    # In altri termini, si cerca di stimare quanto frequentemente le feature latenti siano associate a un singolo token, ovvero se l'attivazione di una feature “dipende” fondamentalmente da un solo token
    split_indices = torch.cumsum(counts, dim=0)
    activation_splits = torch.tensor_split(sorted_activations, split_indices[:-1])
    token_splits = torch.tensor_split(sorted_tokens, split_indices[:-1])

    # This might take a while and we may only care for statistics
    # but for now we do the full loop
    num_single_token_features = 0
    maybe_single_token_features = 0
    for _latent_idx, activation_group, token_group in zip(
        unique_latents, activation_splits, token_splits
    ):
        maybe_single_token, single_token = check_single_feature( activation_group, token_group )
        # maybe_single_token rappresenta una stima “tentativa” o debole che quella feature possa essere considerata come attivata da un singolo token
        # single_token rappresenta una stima più “forte” o sicura che la feature in questione sia effettivamente associata a un singolo token
        
        num_single_token_features += single_token
        # numero totale di feature che sono fortemente considerate single token

        maybe_single_token_features += maybe_single_token 
        # numero di feature che, in una prima stima, sembrano essere “single token” (almeno in modo parziale o incerto)

    single_token_fraction = maybe_single_token_features / num_alive
    strong_token_fraction = num_single_token_features / num_alive
    if verbose:
        print(f"Fraction of weak single token latents: {single_token_fraction:%}")
        print(f"Fraction of strong single token latents: {strong_token_fraction:%}")

    return CacheStatistics(
        frac_alive=fraction_alive,
        frac_fired_1pct=one_percent,
        frac_fired_10pct=ten_percent,
        frac_weak_single_token=single_token_fraction,
        frac_strong_single_token=strong_token_fraction,
    )


@torch.inference_mode()
def check_single_feature(activation_group, token_group):
    sorted_activation_group, sorted_indices = activation_group.sort()
    sorted_token_group = token_group[sorted_indices]

    number_activations = sorted_activation_group.shape[0]
    # Get the first 50 elements if possible
    num_elements = min(50, number_activations)

    wanted_tokens = sorted_token_group[:num_elements]

    # Check how many of them are exactly the same
    _, unique_counts = torch.unique_consecutive(wanted_tokens, return_counts=True)

    max_count = unique_counts.max()
    maybe_single_token = False
    if max_count > 0.9 * num_elements:
        # Single token feature
        maybe_single_token = True

    # Randomly sample 100 activations from the top 50%
    n_top = max(1, int(number_activations * 0.5))
    num_samples = min(100, n_top)
    top_50_percent = sorted_token_group[:n_top]
    sampled_indices = torch.randperm(top_50_percent.shape[0])[:num_samples]
    sampled_tokens = top_50_percent[sampled_indices]
    _, unique_counts = torch.unique_consecutive(sampled_tokens, return_counts=True)

    max_count = unique_counts.max()
    other_maybe_single_token = max_count > 0.75 * num_samples
    if other_maybe_single_token and maybe_single_token:
        return 0, 1
    elif maybe_single_token or other_maybe_single_token:
        return 1, 0
    else:
        return 0, 0
