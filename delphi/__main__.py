import asyncio
import os
from functools import partial
from pathlib import Path
from typing import Callable

import orjson
import torch
from simple_parsing import ArgumentParser
from torch import Tensor
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from delphi.clients import Offline, OpenRouter
from delphi.config import RunConfig
from delphi.explainers import ContrastiveExplainer, DefaultExplainer
from delphi.latents import LatentCache, LatentDataset
from delphi.latents.neighbours import NeighbourCalculator
from delphi.log.result_analysis import log_results
from delphi.pipeline import Pipe, Pipeline, process_wrapper
from delphi.scorers import DetectionScorer, FuzzingScorer
from delphi.sparse_coders import load_hooks_sparse_coders, load_sparse_coders
from delphi.utils import assert_type, load_tokenized_data


# Loads the model and its artifacts
def load_artifacts(run_cfg: RunConfig):

    # Determines the optimal data type for model weights based on configuration and hardware support
    if run_cfg.load_in_8bit:
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = "auto"

    # Loads a pre-trained model using Hugging Face's AutoModel
    model = AutoModel.from_pretrained(
        run_cfg.model,
        device_map={"": "cuda"}, # Places the model on GPU
        quantization_config=(
            BitsAndBytesConfig(load_in_8bit=run_cfg.load_in_8bit)
            if run_cfg.load_in_8bit else None ), 
        torch_dtype=dtype,  # Sets the data type for model weights
        token=run_cfg.hf_token, # Hugging Face API token for accessing models
    )

    # Loads sparse autoencoders (SAEs) for different hookpoints in the model
    hookpoint_to_sparse_encode, transcode = load_hooks_sparse_coders(
        model,
        run_cfg,
        compile=True, # Enables PyTorch compilation for better performance
    )
    """ 
    hookpoint_to_sparse_encode: Dictionary of sparse "encoders" for each hookpoint.
    transcode: When the model's output format needs to be converted to match 
               the sparse autoencoders' input requirements
               (It allows for compatibility between different parts of the system).
               The load_hooks_sparse_coders function determines if transcoding is needed.
    """

    return run_cfg.hookpoints, hookpoint_to_sparse_encode, model, transcode


def create_neighbours(
    run_cfg: RunConfig,
    latents_path: Path,
    neighbours_path: Path,
    hookpoints: list[str],
):
    """
    Creates a neighbours file for the given hookpoints.
    """
    neighbours_path.mkdir(parents=True, exist_ok=True)

    constructor_cfg = run_cfg.constructor_cfg
    saes = (
        load_sparse_coders(run_cfg, device="cpu")
        if constructor_cfg.neighbours_type != "co-occurrence"
        else {}
    )

    for hookpoint in hookpoints:

        if constructor_cfg.neighbours_type == "co-occurrence":
            neighbour_calculator = NeighbourCalculator(
                cache_dir=latents_path / hookpoint, number_of_neighbours=250
            )

        elif constructor_cfg.neighbours_type == "decoder_similarity":

            neighbour_calculator = NeighbourCalculator(
                autoencoder=saes[hookpoint].cuda(), number_of_neighbours=250
            )

        elif constructor_cfg.neighbours_type == "encoder_similarity":
            neighbour_calculator = NeighbourCalculator(
                autoencoder=saes[hookpoint].cuda(), number_of_neighbours=250
            )
        else:
            raise ValueError(
                f"Neighbour type {constructor_cfg.neighbours_type} not supported"
            )

        neighbour_calculator.populate_neighbour_cache(constructor_cfg.neighbours_type)
        neighbour_calculator.save_neighbour_cache(f"{neighbours_path}/{hookpoint}")


async def process_cache(
    run_cfg: RunConfig,
    latents_path: Path,
    explanations_path: Path,
    scores_path: Path,
    hookpoints: list[str],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    latent_range: Tensor | None,
):
    
    """
    Converts SAE latent activations in on-disk cache in the `latents_path` directory
    to latent explanations in the `explanations_path` directory and explanation
    scores in the `fuzz_scores_path` directory.
    """

    # Creates directories for: Explanations, Fuzz scores Detection scores
    explanations_path.mkdir(parents=True, exist_ok=True)
    fuzz_scores_path = scores_path / "fuzz"
    detection_scores_path = scores_path / "detection"
    fuzz_scores_path.mkdir(parents=True, exist_ok=True)
    detection_scores_path.mkdir(parents=True, exist_ok=True)

    # Creates a dictionary mapping hookpoints to their latent ranges. Used to specify which latents to explain
    if latent_range is None:
        latent_dict = None
    else:
        latent_dict = {
            hook: latent_range for hook in hookpoints
        }  # The latent range to explain


    """
    The LatentDataset will construct lazy loaded buffers that load activations into 
    memory when called as an iterator object. 
    For ease of use with the autointerp pipeline, we have a constructor and sampler: 
    - the constructor defines builds the context windows from the cached activations and tokens
    - the sampler divides these contexts into a training and testing set, used to generate explanations and evaluate them.
    """
    dataset = LatentDataset(
        raw_dir=str(latents_path),
        sampler_cfg=run_cfg.sampler_cfg,
        constructor_cfg=run_cfg.constructor_cfg,
        modules=hookpoints,
        latents=latent_dict,
        tokenizer=tokenizer,
    )


    """
    We currently support using OpenRouter's OpenAI compatible API or running locally with VLLM. 
    Define the client you want to use, then create an explainer from the .explainers module.
    """
    if run_cfg.explainer_provider == "offline":
        client = Offline(
            run_cfg.explainer_model,
            max_memory=0.9,
            # Explainer models context length - must be able to accommodate the longest set of examples
            max_model_len=run_cfg.explainer_model_max_len,
            num_gpus=run_cfg.num_gpus,
            statistics=run_cfg.verbose,
        )

    elif run_cfg.explainer_provider == "openrouter":
        if (
            "OPENROUTER_API_KEY" not in os.environ
            or not os.environ["OPENROUTER_API_KEY"]
        ):
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. Set "
                "`--explainer-provider offline` to use a local explainer model."
            )

        client = OpenRouter(
            run_cfg.explainer_model,
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        
    else:
        raise ValueError(
            f"Explainer provider {run_cfg.explainer_provider} not supported"
        )


    def explainer_postprocess(result):
        with open(explanations_path / f"{result.record.latent}.txt", "wb") as f:
            f.write(orjson.dumps(result.explanation))

        return result

    if run_cfg.constructor_cfg.non_activating_source == "FAISS":
        explainer = ContrastiveExplainer(
            client,
            threshold=0.3,
            verbose=run_cfg.verbose,
        )
    else:
        explainer = DefaultExplainer(
            client,
            threshold=0.3,
            verbose=run_cfg.verbose,
        )

    
    """
    The explainer should be added to a pipe, which will send the explanation requests to the client. 
    The pipe should have a function that happens after the request is completed, 
    to e.g. save the data, and could also have a function that happens before the request is sent, 
    e.g to transform some of the data.
    """
    explainer_pipe = Pipe(process_wrapper(explainer, postprocess=explainer_postprocess))
    """
    The pipe should then be used in a pipeline. Running the pipeline will send requests to the client 
    in batches of paralel requests.
    """


    # Builds the record from result returned by the pipeline
    def scorer_preprocess(result):
        if isinstance(result, list):
            result = result[0]

        record = result.record
        record.explanation = result.explanation
        record.extra_examples = record.not_active
        return record

    # Saves the score to a file
    def scorer_postprocess(result, score_dir):
        safe_latent_name = str(result.record.latent).replace("/", "--")

        with open(score_dir / f"{safe_latent_name}.txt", "wb") as f:
            f.write(orjson.dumps(result.score))

    scorer_pipe = Pipe(
        process_wrapper(
            DetectionScorer(
                client,
                n_examples_shown=run_cfg.num_examples_per_scorer_prompt,
                verbose=run_cfg.verbose,
                log_prob=False,
            ),
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir=detection_scores_path),
        ),
        process_wrapper(
            FuzzingScorer(
                client,
                n_examples_shown=run_cfg.num_examples_per_scorer_prompt,
                verbose=run_cfg.verbose,
                log_prob=False,
            ),
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir=fuzz_scores_path),
        ),
    )


    """
    FINAL PIPELINE
    """
    pipeline = Pipeline(
        dataset,
        explainer_pipe,
        scorer_pipe,
    )

    await pipeline.run(run_cfg.pipeline_num_proc)


def populate_cache(
    run_cfg: RunConfig,
    model: PreTrainedModel,
    hookpoint_to_sparse_encode: dict[str, Callable],
    latents_path: Path,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    transcode: bool,
):
    """
    Populates an on-disk cache in `latents_path` with SAE latent activations.
    """
    latents_path.mkdir(parents=True, exist_ok=True)

    # Create a log path within the run directory
    log_path = latents_path.parent / "log"
    log_path.mkdir(parents=True, exist_ok=True)

    cache_cfg = run_cfg.cache_cfg
    tokens = load_tokenized_data(
        cache_cfg.cache_ctx_len,
        tokenizer,
        cache_cfg.dataset_repo,
        cache_cfg.dataset_split,
        cache_cfg.dataset_name,
        cache_cfg.dataset_column,
        run_cfg.seed,
    )

    if run_cfg.filter_bos:
        if tokenizer.bos_token_id is None:
            print("Tokenizer does not have a BOS token, skipping BOS filtering")
        else:
            flattened_tokens = tokens.flatten()
            mask = ~torch.isin(flattened_tokens, torch.tensor([tokenizer.bos_token_id]))
            masked_tokens = flattened_tokens[mask]
            truncated_tokens = masked_tokens[
                : len(masked_tokens) - (len(masked_tokens) % cache_cfg.cache_ctx_len)
            ]
            tokens = truncated_tokens.reshape(-1, cache_cfg.cache_ctx_len)


    """
    The first step to generate explanations is to cache sparse model activations. 
    To do so, load your sparse models into the base model, load the tokens you want 
    to cache the activations from, create a LatentCache object and run it.
    """
    cache = LatentCache(
        model,
        hookpoint_to_sparse_encode,
        batch_size=cache_cfg.batch_size,
        transcode=transcode,
        log_path=log_path,
    )
    cache.run(cache_cfg.n_tokens, tokens) # Processes tokens in batches to generate latent activations using sparse encoders at specified hookpoints

    """
    The second step is to save the splits to the latents path.
    Caching saves .safetensors of dict["activations", "locations", "tokens"]
    Safetensors are split into shards over the width of the autoencoder.
    """
    if run_cfg.verbose:
        cache.generate_statistics_cache()

    cache.save_splits(
        # Split the activation and location indices into different files to make
        # loading faster
        n_splits=cache_cfg.n_splits,
        save_dir=latents_path,
    )
    cache.save_config(save_dir=latents_path, cfg=cache_cfg, model_name=run_cfg.model)


def non_redundant_hookpoints(
    hookpoint_to_sparse_encode: dict[str, Callable] | list[str],
    results_path: Path,
    overwrite: bool,
) -> dict[str, Callable] | list[str]:
    """
    Returns a list of hookpoints that are not already in the cache.
    """
    if overwrite:
        print("Overwriting results from", results_path)
        return hookpoint_to_sparse_encode
    in_results_path = [x.name for x in results_path.glob("*")]
    if isinstance(hookpoint_to_sparse_encode, dict):
        non_redundant_hookpoints = {
            k: v
            for k, v in hookpoint_to_sparse_encode.items()
            if k not in in_results_path
        }
    else:
        non_redundant_hookpoints = [
            hookpoint
            for hookpoint in hookpoint_to_sparse_encode
            if hookpoint not in in_results_path
        ]
    if not non_redundant_hookpoints:
        print(f"Files found in {results_path}, skipping...")
    return non_redundant_hookpoints


async def run(
    run_cfg: RunConfig,
):
    
    """
    Creates a base directory for results.
    If a specific name is provided in the config, creates a subdirectory with that name.
    Creates all necessary parent directories
    """
    base_path = Path.cwd() / "results"
    if run_cfg.name:
        base_path = base_path / run_cfg.name
    base_path.mkdir(parents=True, exist_ok=True)


    # Saves the run configuration to a JSON file
    run_cfg.save_json(base_path / "run_config.json", indent=4)

    # Sets up paths for different types of outputs:
    latents_path = base_path / "latents" # For storing model activations
    explanations_path = base_path / "explanations" # For storing explanation of the latents
    scores_path = base_path / "scores" # For storing evaluation scores
    neighbours_path = base_path / "neighbours" # For storing neighbor relationships
    visualize_path = base_path / "visualize" # For storing visualization data


    # Creates a range of latents to analyze (if specified)
    latent_range = torch.arange(run_cfg.max_latents) if run_cfg.max_latents else None

    # Loads the model and its artifacts
    hookpoints, hookpoint_to_sparse_encode, model, transcode = load_artifacts(run_cfg)

    # Initializes the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(run_cfg.model, token=run_cfg.hf_token)


    # This code determines which hookpoints need to be processed by checking which ones haven't been processed yet
    nrh = assert_type(
        dict,
        non_redundant_hookpoints(
            hookpoint_to_sparse_encode, latents_path, "cache" in run_cfg.overwrite
        ),
    )

    if nrh: # proceeds if there are non-redundant hookpoints to process
        populate_cache( # Populates an on-disk cache with SAE (Sparse Autoencoder) latent activations
            run_cfg,
            model,
            nrh, # The hookpoints that need processing
            latents_path, # Where to save the results
            tokenizer, # For processing text input
            transcode,
        )

    del model, hookpoint_to_sparse_encode 
    # Frees up memory by deleting the model and sparse encoders 
    # These are no longer needed after cache population

    if run_cfg.constructor_cfg.non_activating_source == "neighbours":
        nrh = assert_type(
            list,
            non_redundant_hookpoints(
                hookpoints, neighbours_path, "neighbours" in run_cfg.overwrite
            ),
        )
        if nrh:
            create_neighbours(
                run_cfg,
                latents_path,
                neighbours_path,
                nrh,
            )
    else:
        print("Skipping neighbour creation")

    # Checks which hookpoints need score processing. Avoids reprocessing already scored hookpoints.
    nrh = assert_type(
        list,
        non_redundant_hookpoints(
            hookpoints, scores_path, "scores" in run_cfg.overwrite
        ),
    )

    """
    If there are hookpoints needing processing:
    - Processes the cached activations
    - Generates explanations
    - Calculates scores
    - Runs asynchronously for better performance
    """
    if nrh:
        await process_cache(
            run_cfg,
            latents_path,
            explanations_path,
            scores_path,
            nrh,
            tokenizer,
            latent_range,
        )


    if run_cfg.verbose:
        log_results(scores_path, visualize_path, run_cfg.hookpoints)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(RunConfig, dest="run_cfg") # Adds all the configuration options from the RunConfig class to the parser
    args = parser.parse_args()

    asyncio.run(run(args.run_cfg))
