import sys
import json
import pickle
import os
import zlib
import base64
import re
from typing import List, Dict, Tuple, Any, Optional, Union
import time
import math
import subprocess
import shutil
from os.path import dirname, abspath, join
import multiprocessing
from functools import partial
from pympler import asizeof

import pathlib
import toml
from dynaconf import Dynaconf
from jinja2 import Template
from loguru import logger
from tqdm import tqdm

def extract_code(text: str, language: str = "auto",
                 allow_no_language_label: bool = True, verbose: bool = True,
                 return_code_list: bool = False, raise_exception: bool = True) -> Union[str, List[str], None]:
    """
    Extract code from text.

    Args:
        text (str): The text to extract code from.
        language (str): The language of the code to extract. If 'auto', all languages will be tried.
        allow_no_language_label (bool): Whether to allow no language label. If True, the code will be extracted regardless of the language label.
        verbose (bool): Whether to print verbose information.
        return_code_list (bool): Whether to return a list of codes. If False, only the longest code will be returned.
        raise_exception (bool): Whether to raise an exception if the code is not found. If False, None will be returned when the code is not found.

    Returns:
        Union[str, List[str], None]: The extracted code. If return_code_list is True, a list of codes will be returned.
    """
    if text is None:
        if raise_exception:
            raise ValueError("Text is None.")
        else:
            return [] if return_code_list else None
    code_list = []
    
    language_to_patterns = {
        'python': [r"```python(.*?)```"],
        'cpp': [r"```cpp(.*?)```", r"```c\+\+(.*?)```"],
        'json': [r"```json(.*?)```"],
    }
    if language in language_to_patterns:
        patterns = language_to_patterns[language]
        for pattern in patterns:
            code_list += re.findall(pattern, text, re.DOTALL)
    elif language == 'auto':
        for lan in language_to_patterns:
            for pattern in language_to_patterns[lan]:
                code_list += re.findall(pattern, text, re.DOTALL)
    else:
        if raise_exception:
            raise ValueError(f"Unsupported language: {language}")
        else:
            return [] if return_code_list else None
    
    if len(code_list) == 0:
        if not allow_no_language_label:
            if raise_exception:
                raise Exception("Failed to extract code.")
            else:
                return [] if return_code_list else None
        new_pattern = r"```(.*?)```"
        if verbose:
            logger.warning(f"Failed to extract code. Retry with a more general pattern: {new_pattern}.")
        code_list += re.findall(new_pattern, text, re.DOTALL)
        if len(code_list) == 0:
            if raise_exception:
                raise Exception("Failed to extract code.")
            else:
                return [] if return_code_list else None
        if return_code_list:
            return [i.strip() for i in code_list]
        else:
            code = max(code_list, key=len)
            return code.strip()
    if return_code_list:
        return [i.strip() for i in code_list]
    else:
        code = max(code_list, key=len)
        if code != code_list[-1] and verbose:
            logger.warning(f"The longest code is not the last one. There might be some errors.")
        return code.strip()

def encode_testcases(testcases: List[Dict[str, str]]) -> str:
    """
    According to LiveCodeBench, private test cases should be encoded.
    """
    json_str = json.dumps(testcases)
    pickled_data = pickle.dumps(json_str)
    compressed_data = zlib.compress(pickled_data)
    encoded_testcases = base64.b64encode(compressed_data).decode('utf-8')
    return encoded_testcases

def decode_testcases(encoded_testcases: str) -> List[Dict[str, str]]:
    return json.loads(
        pickle.loads(
            zlib.decompress(
                base64.b64decode(encoded_testcases.encode("utf-8"))
            )
        )
    )

def make_dir_for_file(file_path: str) -> None:
    """
    Make directory for file_path
    """
    dir_name = os.path.dirname(file_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def save_json(data: Any, file_path: str) -> None:
    """
    Save data to file_path with json format
    """
    make_dir_for_file(file_path)
    with open(file_path, "w") as f:
        json.dump(data, f)
        
def load_json(file_path: str) -> Any:
    """
    Load data from file_path with json format
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def load_json_line(file_path: str, use_eval: bool = False, verbose: bool = False) -> List[Any]:
    """
    Load data from file_path with json line (i.e., regard each line as a json file) format
    """
    data = []
    with open(file_path, "r") as f:
        for line in tqdm(f, disable=not verbose):
            try:
                if use_eval:
                    data.append(eval(line))
                else:
                    data.append(json.loads(line))
            except:
                if verbose:
                    print(f'[ERROR] broken line: {line[:20]}')
                continue
    return data

def save_json_line(data: List[Any], file_path: str, do_append: bool=False) -> None:
    """
    Save data to file_path with json line (i.e., regard each line as a json file) format
    """
    make_dir_for_file(file_path)
    mode = 'a' if do_append else 'w'
    with open(file_path, mode) as f:
        for d in data:
            f.write(json.dumps(d) + '\n')

def save_pickle(data: Any, file_path: str) -> None:
    """
    Save data from file_path with pickle format
    """
    make_dir_for_file(file_path)
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

def load_pickle(file_path: str) -> Any:
    """
    Load data from file_path with pickle format
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data
        
def force_delete_folder(folder_path: str) -> None:
    """
    Force delete a folder and all its contents, including files that may be locked by processes.
    """
    if not os.path.exists(folder_path):
        logger.warning(f"Folder {folder_path} does not exist.")
        return
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file.startswith(".nfs"):
                    result = subprocess.run(["lsof", file_path], capture_output=True, text=True)
                    if result.stdout:
                        for line in result.stdout.strip().split("\n")[1:]:
                            pid = int(line.split()[1])
                            subprocess.run(["kill", "-9", str(pid)])
                os.unlink(file_path)
                # logger.debug(f"{file_path} has been deleted.")
            except Exception as e:
                logger.warning(f"Falied to delete file {file_path} with os.unlink. Error: {e}")
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            try:
                shutil.rmtree(dir_path)
            except Exception as e:
                logger.warning(f"Falied to delete sub-folder {dir_path} with shutil.rmtree. Error: {e}")
    try:
        shutil.rmtree(folder_path)
    except Exception as e:
        logger.warning(f"Falied to delete folder {folder_path} with shutil.rmtree. Error: {e}")
    try:
        subprocess.run(["rm", "-rf", folder_path], check=False)
    except Exception as e:
        logger.warning(f"Falied to delete folder {folder_path} with rm -rf. Error: {e}")
    
    if os.path.exists(folder_path):
        logger.error(f"Failed to delete folder {folder_path}.")
    else:
        # logger.debug(f"Folder {folder_path} has been deleted successfully.")
        pass

def load_prompt_templates(current_dir: str):
    """
    Load prompt templates from the './prompt_templates' directory.
    """
    current_dir = dirname(abspath(current_dir))
    template_dir = os.path.join(current_dir, 'prompt_templates')
    toml_files = list(pathlib.Path(join(template_dir)).glob('*.toml'))
    prompt_templates = Dynaconf(
        envvar_prefix=False,
        merge_enabled=True,
        settings_files=toml_files,
    )
    return prompt_templates


def get_most_reliable_solutions(
    pdata: Dict[str, Any],
    language_list: Optional[List[str]]=None,
    lowest_reliability_level: int = -1,
    max_num: int = 100
) -> List[Dict[str, Any]]:
    """
    Get the most reliable solutions from the problem data.
    
    Args:
        pdata (Dict[str, Any]): The problem data.
        language_list (List[str], optional): The list of languages to use. Defaults to None.
        lowest_reliability_level (int, optional): The lowest reliability level of the solutions to use. Defaults to -1.
        max_num (int, optional): The maximum number of solutions to return. Defaults to 100.
    
    Returns:
        List[Dict[str, Any]]: The most reliable solutions. Fields: 
            code, language, source, reliability_level, code_solution_id.
    """
    language_set = None if language_list is None else set(language_list)
    language_order = {}
    if language_list is not None:
        language_order = {lang: idx for idx, lang in enumerate(language_list)}
    
    pid = pdata['pid']
    sol_list = []
    for idx, sol_data in enumerate(pdata['solutions']):
        source = sol_data['source']
        code = sol_data['code']
        language = sol_data['language']
        if language_set is not None and language not in language_set:
            continue
        reliability_level_dict = {
            'atcoder_submission': 5,
            'editorial': 5, # editorial from Jack Zeng
            'code_contests': 4,
            'luogu_editorial': 3, # editorial from Luogu
            'taco-verified': 2,
            'taco': 1,
        }
        reliability_level = reliability_level_dict[source]
        if source == 'atcoder_submission' and language == 'python':
            reliability_level = 1 # atcoder_submission of python may use a customized package called "atcoder"
        if reliability_level < lowest_reliability_level:
            continue
        sol_list.append({
            'code': code, 'language': language, 'source': source,
            'reliability_level': reliability_level,
            'code_solution_id': f'{pid}_{idx}'
        })

    def sort_key(x):
        lang_idx = language_order.get(x['language'], len(language_order))
        return (x['reliability_level'], -lang_idx)
    
    return sorted(sol_list, key=sort_key, reverse=True)[:max_num]

def flatten_data_dict(data_dict):
    mapping = dict()
    result_data_list = []
    for data_type in data_dict:
        if len(data_dict[data_type]) == 0:
            mapping[data_type] = []
            continue
        if isinstance(data_dict[data_type][0], list):
            mapping[data_type] = []
            for data_ls in data_dict[data_type]:
                mapping[data_type].append([])
                for data_item in data_ls:
                    mapping[data_type][-1].append(len(result_data_list))
                    result_data_list.append(data_item)
        else:
            mapping[data_type] = []
            for data_item in data_dict[data_type]:
                mapping[data_type].append(len(result_data_list))
                result_data_list.append(data_item)
            
    return mapping, result_data_list

def unflatten_data_dict(mapping, result_data_list):
    data_dict = {}
    for data_type, idxs in mapping.items():
        if len(idxs) == 0:
            data_dict[data_type] = []
        else:
            if all(isinstance(sub, list) for sub in idxs):
                data_dict[data_type] = [
                    [result_data_list[i] for i in sub] for sub in idxs
                ]
            else:
                data_dict[data_type] = [result_data_list[i] for i in idxs]
    return data_dict

def remove_none_from_test_cases(
    mapping,
    test_case_list
):
    tc_dict = unflatten_data_dict(mapping, test_case_list)
    tc_dict_filtered = dict()
    for tc_type in tc_dict:
        if len(tc_dict[tc_type]) == 0:
            tc_dict_filtered[tc_type] = []
            continue
        if isinstance(tc_dict[tc_type][0], list):
            tc_dict_filtered[tc_type] = [
                [tc for tc in tc_list if tc['input'] is not None and tc['output'] is not None]
            for tc_list in tc_dict[tc_type]]
        else:
            tc_dict_filtered[tc_type] = [tc for tc in tc_dict[tc_type] if tc['input'] is not None and tc['output'] is not None]
    return flatten_data_dict(tc_dict_filtered)

def get_object_size_mb(obj: Any) -> float:
    """
    Get the size of an object in megabytes.
    """
    size_bytes = asizeof.asizeof(obj)
    return size_bytes / (1024 * 1024)