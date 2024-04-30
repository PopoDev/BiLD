from transformers import PreTrainedModel, PreTrainedTokenizerBase
from datasets import DatasetDict
from typing import Dict, List
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import os
import numpy as np
import sys

def getEmptyFrameDict(
    input_key: str,
    output_key: str,
    dataset_dict: DatasetDict
) -> Dict[str, pd.DataFrame]:
    dataframe_dict = dict()

    for curr_name, curr_dataset in dataset_dict.items():
        next_dataframe = pd.DataFrame.from_dict({
            input_key: curr_dataset[input_key],
            output_key: [""] * len(curr_dataset[input_key])
        })

        dataframe_dict[curr_name] = next_dataframe

    return dataframe_dict

def cacheFrame(
    data_frame: pd.DataFrame,
    cache_dir: Path,
    cache_entry_name: str
) -> None:
    file_name = f"{cache_entry_name}.cache.parquet"
    data_frame.to_parquet(
        os.path.join(
            cache_dir,
            file_name
        ),
        engine = "pyarrow",
        compression = None
    )

def cacheFrameDict(
    cache_dir: Path,
    dataframe_dict: Dict[str, pd.DataFrame],
    prefix_name: str = None,
) -> None:
    for curr_name, curr_dataframe in dataframe_dict.items():
        file_name = curr_name

        if prefix_name is not None:
            file_name = f"{prefix_name}_{file_name}"

        cacheFrame(
            curr_dataframe,
            cache_dir,
            file_name
        )

def loadFrameDict(
    cache_dir: Path,
    dataframe_names: List[str],
    prefix_name: str = None,
) -> Dict[str, pd.DataFrame]:
    dataframe_dict = dict()

    for curr_name in dataframe_names:
        file_name = f"{curr_name}.cache.parquet"

        if prefix_name is not None:
            file_name = f"{prefix_name}_{file_name}"

        dataframe_dict[curr_name] = pd.read_parquet(
            os.path.join(
                cache_dir,
                file_name
            ),
            engine = "pyarrow",
        )

    return dataframe_dict

def inferDataFrameDict(
    dataframe_dict: Dict[str, pd.DataFrame],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    input_key: str = "article",
    output_key: str = "t5_large_output",
    prefix: str = "summarize: ",
    max_input_length: int = 512,
    max_output_length: int = 512,
    cache_location: str = None,
    dataset_name: str = None,
    batches_per_cache_write: int = None
) -> None:
    using_cache = all(x is not None for x in [cache_location, dataset_name, batches_per_cache_write])

    if model.training:
        model.eval()

    model_device = model.device

    for curr_name, curr_dataframe in dataframe_dict.items():
        chunks_iter = np.array_split(curr_dataframe, (len(curr_dataframe) // batch_size) + 1)
        row_counter = 0
        
        for chunk_idx, curr_chunk in enumerate(tqdm(chunks_iter)):
            try:
                curr_chunk_inputs = list(curr_chunk[input_key])
                curr_chunk_outputs = list(curr_chunk[output_key])
                is_cached = all(x != "" for x in curr_chunk_outputs)

                if is_cached:
                    row_counter += len(curr_chunk_outputs)
                    continue

                inputs = [prefix + doc for doc in curr_chunk_inputs]
                input_ids = tokenizer(
                    inputs, 
                    return_tensors = "pt",
                    max_length = max_input_length,
                    truncation = True,
                    padding = True,
                ).input_ids.to(model_device)

                outputs = model.generate(input_ids, max_new_tokens = max_output_length)
                outputs.to("cpu")
                decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens = True)

                print(decoded_output)

                out_column_index = curr_dataframe.columns.get_loc(output_key)
                end_row_index = row_counter + len(decoded_output)
                curr_dataframe.iloc[row_counter : end_row_index, out_column_index] = decoded_output
                row_counter += len(curr_chunk_outputs)

                if (((chunk_idx + 1) % batches_per_cache_write) == 0) and using_cache:
                    cacheFrame(
                        curr_dataframe,
                        cache_location,
                        f"{dataset_name}_{curr_name}"
                    )
            except:
                sys.exit(1)
            
        cacheFrame(
            curr_dataframe,
            cache_location,
            f"{dataset_name}_{curr_name}"
        )

# Hugging Face doesn't provide an easy way to load a DatasetDict from Pandas,
# so loadDatasetFromCachedDataframe pulls the latest cache entry instead
def loadDatasetFromCachedDataframe(
    cache_dir: Path,
    dataframe_names: List[str],
    prefix_name: str = None,
) -> DatasetDict:
    file_dict = dict()

    for curr_name in dataframe_names:
        file_name = f"{curr_name}.cache.parquet"

        if prefix_name is not None:
            file_name = f"{prefix_name}_{file_name}"

        file_dict[curr_name] = os.path.join(cache_dir, file_name)

    return DatasetDict.from_parquet(file_dict)
