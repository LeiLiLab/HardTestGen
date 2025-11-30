import json
import os
import sys
import time
import yaml
import asyncio
from pprint import pformat, pprint
import random
import re
from typing import List, Dict, Any, Tuple, Optional, Union, Set
from pprint import pprint
from collections import defaultdict, Counter
import pathlib
import toml
import ast
import argparse
from pathlib import Path

from tqdm import tqdm
from loguru import logger
import openai
import nest_asyncio
import json
import pandas as pd
import datasets
from jinja2 import Template
from dynaconf import Dynaconf
from tenacity import retry, stop_after_attempt, wait_random_exponential

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from hard_tests_gen import utils
from hard_tests_gen.llm_api import get_completion_for_prompts

prompt_templates = utils.load_prompt_templates(os.path.join(os.path.dirname(__file__), "prompt_templates"))

def get_iv_and_ojf_gen_prompt(
    question_content: str,
    oracle_program: str,
    prompt_template: str = 'test_cases_kit_prompt_iv_and_ojf'
):
    template = Template(prompt_templates[prompt_template]['content'])
    prompt = template.render(
        problem_specification=question_content,
        oracle_program=oracle_program
    )
    return prompt.strip()

def get_ig_gen_prompt(
    question_content: str,
    oracle_program: str,
    input_validator: str,
    num_LLMGen_input: int = 10,
    prompt_template: str = 'test_cases_kit_prompt_ig',
):
    template = Template(prompt_templates[prompt_template]['content'])
    prompt = template.render(
        num_LLMGen_input=str(num_LLMGen_input),
        problem_specification=question_content,
        oracle_program=oracle_program,
        input_validator=input_validator
    )
    return prompt.strip()

def _fix_control_chars_inside_strings(s: str) -> str:
    out = []
    in_str = False
    escape = False

    for ch in s:
        if in_str:
            if escape:
                out.append(ch)
                escape = False
            else:
                if ch == '\\':
                    out.append(ch)
                    escape = True
                elif ch == '"':
                    out.append(ch)
                    in_str = False
                else:
                    c = ord(ch)
                    if c < 0x20:
                        if ch == '\n':
                            out.append('\\n')
                        elif ch == '\r':
                            out.append('\\r')
                        elif ch == '\t':
                            out.append('\\t')
                        else:
                            out.append('\\u%04x' % c)
                    else:
                        out.append(ch)
        else:
            if ch == '"':
                in_str = True
                out.append(ch)
            else:
                out.append(ch)

    return ''.join(out)


def parse_test_cases_kit_response(response: str) -> Dict[str, Any]:
    try:
        match = re.search(r'(?<=# Result)[\s\S]*', response)
        if not match:
            raise ValueError("No '# Result' section found in response")
        result_content = match.group().strip()

        json_str = utils.extract_code(text=result_content, language='json', allow_no_language_label=False, verbose=False)
        if json_str is None:
            raise ValueError("utils.extract_code returned None")

        json_str = json_str.strip()
        try:
            return json.loads(json_str)
        except:
            json_fixed = _fix_control_chars_inside_strings(json_str)
            return json.loads(json_fixed)
    except:
        return dict()


def get_test_cases_kit_info_msg(test_cases_kit) -> str:
    num_RPGen_SPGen_input_func_names = len(test_cases_kit['input_generation']['RPGen_SPGen_input_generator']['func_names'])
    num_HackGen_input_func_names = len(test_cases_kit['input_generation']['HackGen_input_generator']['func_names'])
    num_LLMGen_input = len(test_cases_kit['input_generation']['LLMGen_input'])
    info = [
        f"LLMGen_input ({num_LLMGen_input})"
        f"RPGen_SPGen_input ({num_RPGen_SPGen_input_func_names})"
        f"HackGen_input ({num_HackGen_input_func_names})"
        f"IV {'(√)' if test_cases_kit['input_validator'] is not None else '(x)'}",
        f"OJF {'(√)' if test_cases_kit['output_judging_function'] is not None else '(x)'}"
    ]
    return ', '.join(info)


def extract_test_cases_kit(pid: str, iv_and_ojf_gen_response: str, ig_gen_response: str) -> Dict[str, Any]:    
    iv_and_ojf_gen_dict = parse_test_cases_kit_response(iv_and_ojf_gen_response)
    input_validator = iv_and_ojf_gen_dict.get('input_validator', None)
    output_judging_function = iv_and_ojf_gen_dict.get('output_judging_function', None)
    
    ig_gen_dict = parse_test_cases_kit_response(ig_gen_response)
    RPGen_SPGen_input_code = ig_gen_dict.get('RPGen_SPGen_input_generator', None)
    RPGen_SPGen_input_func_names = get_function_names(RPGen_SPGen_input_code)
    RPGen_SPGen_input_func_names = [name for name in RPGen_SPGen_input_func_names if name.startswith('gen_stratified_input') or name.startswith('gen_range_based_input')]
    HackGen_input_code = ig_gen_dict.get('HackGen_input_generator', None)
    HackGen_input_func_names = get_function_names(HackGen_input_code)
    HackGen_input_func_names = [name for name in HackGen_input_func_names if name.startswith('gen_hacking_input')]
    LLMGen_input_list = ig_gen_dict.get('LLMGen_input', [])
    LLMGen_input_list = [str(d) for d in LLMGen_input_list]
    
    test_cases_kit = {
        'pid': pid,
        'input_generation': {
            'LLMGen_input': LLMGen_input_list,
            'RPGen_SPGen_input_generator': {
                'code': RPGen_SPGen_input_code,
                'func_names': RPGen_SPGen_input_func_names,
            },
            'HackGen_input_generator': {
                'code': HackGen_input_code,
                'func_names': HackGen_input_func_names,
            }
        },
        'input_validator': input_validator,
        'output_judging_function': output_judging_function,
        'original_responses':{
            'iv_and_ojf_generation_response': iv_and_ojf_gen_response,
            'ig_generation_response': ig_gen_response,
        }
    }
    return test_cases_kit

def get_function_names(code_str: str) -> List[str]:
    try:
        if code_str is None:
            return []
        tree = ast.parse(code_str)
        function_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_names.append((node.lineno, node.name))
        function_names.sort()
        return [name for _, name in function_names]
    except Exception as e:
        print(e)
        return []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name_or_path', type=str)
    parser.add_argument('--target_pids_path', type=str, default=None, help='Path to the JSON file containing target problem IDs. If not provided, all problems in the dataset will be used.')
    parser.add_argument('--iv_and_ojf_gen_prompt_template', type=str, default='test_cases_kit_prompt_iv_and_ojf', help='Prompt template for Input Validator (IV) and Output Judging Function (OJF) generation.')
    parser.add_argument('--ig_gen_prompt_template', type=str, default='test_cases_kit_prompt_ig', help='Prompt template for Input Generation (IG) generation')
    parser.add_argument('--num_LLMGen_input', type=int, default=10, help='Number of LLMGen inputs to generate.')
    parser.add_argument('--iv_and_ojf_gen_responses_save_path', type=str, default='./iv_and_ojf_gen_responses.jsonl', help='Path to save Input Validator (IV) and Output Judging Function (OJF) generation responses.')
    parser.add_argument('--ig_gen_responses_save_path', type=str, default='./ig_gen_responses.jsonl', help='Path to save Input Generation (IG) generation responses.')
    parser.add_argument('--test_cases_kit_save_path', type=str, default='./test_cases_kits.jsonl', help='Path to save the generated test cases kits')
    parser.add_argument('--model_name', type=str, default='gpt-4o', help='Name of LLM to use for generation.')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for the LLM.')
    parser.add_argument('--max_tokens', type=int, default=5120, help='Maximum number of tokens for the LLM.')
    parser.add_argument('--num_parallel', type=int, default=100, help='Number of parallel requests to the LLM.')
    args = parser.parse_args()
    logger.info(f"Arguments: {pformat(vars(args))}")
    return args

def main():
    args = parse_args()
    
    if os.path.exists(args.dataset_name_or_path):
        dataset_path = Path(args.dataset_name_or_path)
        if dataset_path.suffix == '.json':
            pdata_list = utils.load_json(args.dataset_name_or_path)
        elif dataset_path.suffix == '.jsonl':
            pdata_list = utils.load_json_line(args.dataset_name_or_path)
        else:
            raise ValueError(f"Unsupported dataset file format: {dataset_path.suffix}")
    else:
        pdata_list = datasets.load_dataset(args.dataset_name_or_path)['train'].to_list()
    pid_to_pdata = {pdata['pid']: pdata for pdata in pdata_list}
    if args.target_pids_path:
        target_pids = utils.load_json(args.target_pids_path)
    else:
        target_pids = [pdata['pid'] for pdata in pdata_list]
    
    logger.info(f"Num total problems: {len(pdata_list)}")
    logger.info(f"Num target problems: {len(target_pids)}")
    
    # Generate Input Validator (IV) and Output Judging Function (OJF)
    logger.info("Generating Input Validator (IV) and Output Judging Function (OJF) prompts...")
    iv_and_ojf_gen_prompt_data = []
    num_problems_without_oracle_program = 0
    
    for pid in tqdm(target_pids):
        pdata = pid_to_pdata[pid]
        question_content = pdata['question_content']
        oracle_programs = utils.get_most_reliable_solutions(
            pdata=pdata,
            language_list=['cpp', 'python3', 'python'],
            lowest_reliability_level=1,
            max_num=1
        )
        if len(oracle_programs) == 0:
            num_problems_without_oracle_program += 1
            continue
        else:
            oracle_program = oracle_programs[0]['code']
        iv_and_ojf_gen_prompt = get_iv_and_ojf_gen_prompt(
            question_content=question_content,
            oracle_program=oracle_program,
            prompt_template=args.iv_and_ojf_gen_prompt_template
        )
        iv_and_ojf_gen_prompt_data.append({
            'pid': pid,
            'iv_and_ojf_gen_prompt': iv_and_ojf_gen_prompt,
        })
    logger.info(f"Num problems without correct program: {num_problems_without_oracle_program}")
    
    prompt_list = [d['iv_and_ojf_gen_prompt'] for d in iv_and_ojf_gen_prompt_data]
    pid_list = [d['pid'] for d in iv_and_ojf_gen_prompt_data]
    get_completion_for_prompts(
        prompt_list=prompt_list,
        id_list=pid_list,
        model_name=args.model_name,
        num_parallel=args.num_parallel,
        save_path=args.iv_and_ojf_gen_responses_save_path,
        save_steps=20,
        id_field_name='pid',
        responses_field_name='iv_and_ojf_gen_response',
        do_return=False,
        load_finished=True,
        log_steps=1,
        use_litellm=True,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=1,
    )
    
    iv_and_ojf_gen_responses = utils.load_json_line(args.iv_and_ojf_gen_responses_save_path)
    failed_pids = []
    pid_to_iv = dict()
    for response_data in iv_and_ojf_gen_responses:
        response = response_data['iv_and_ojf_gen_response'][0]
        pid = response_data['pid']
        iv_and_ojf_gen_dict = parse_test_cases_kit_response(response)
        if 'input_validator' in iv_and_ojf_gen_dict:
            iv = iv_and_ojf_gen_dict['input_validator']
        else:
            iv = "Failed to generate input validator."
            print(f"pid: {pid}, Failed to generate input validator.")
        pid_to_iv[pid] = iv
        if iv == "Failed to generate input validator.":
            failed_pids.append(pid)
    failed_pids = set(failed_pids)
    iv_and_ojf_gen_responses = [d for d in iv_and_ojf_gen_responses if d['pid'] not in failed_pids]
    utils.save_json_line(iv_and_ojf_gen_responses, args.iv_and_ojf_gen_responses_save_path)
    logger.info(f"Num problems with failed IV generation: {len(failed_pids)}")
    
    # Input Generation
    logger.info("Generating Input Generation (IG) prompts...")
    num_problems_without_oracle_program = 0
    num_problems_without_input_validator = 0

    ig_gen_prompt_data = []
    for pid in tqdm(target_pids):
        pdata = pid_to_pdata[pid]
        question_content = pdata['question_content']
        oracle_programs = utils.get_most_reliable_solutions(
            pdata=pdata,
            language_list=['cpp', 'python3', 'python'],
            lowest_reliability_level=1,
            max_num=1
        )
        if len(oracle_programs) == 0:
            num_problems_without_oracle_program += 1
            continue
        else:
            oracle_program = oracle_programs[0]['code']
        if pid not in pid_to_iv:
            num_problems_without_input_validator += 1
            continue
        else:
            input_validator = pid_to_iv[pid]
        ig_gen_prompt = get_ig_gen_prompt(
            question_content=question_content,
            oracle_program=oracle_program,
            input_validator=input_validator,
            num_LLMGen_input=args.num_LLMGen_input,
            prompt_template= args.ig_gen_prompt_template
        )
        ig_gen_prompt_data.append({
            'pid': pid,
            'ig_gen_prompt': ig_gen_prompt,
        })
    logger.info(f"Num problems without correct program: {num_problems_without_oracle_program}")
    logger.info(f"Num problems without input validator: {num_problems_without_input_validator}")
    
    prompt_list = [d['ig_gen_prompt'] for d in ig_gen_prompt_data]
    pid_list = [d['pid'] for d in ig_gen_prompt_data]
    get_completion_for_prompts(
        prompt_list=prompt_list,
        id_list=pid_list,
        model_name=args.model_name,
        num_parallel=args.num_parallel,
        save_path=args.ig_gen_responses_save_path,
        save_steps=20,
        id_field_name='pid',
        responses_field_name='ig_gen_response',
        do_return=False,
        load_finished=True,
        log_steps=1,
        use_litellm=True,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=1,
    )
    
    # Make test cases kits
    logger.info("Making test cases kits...")
    
    iv_and_ojf_gen_responses = utils.load_json_line(args.iv_and_ojf_gen_responses_save_path)
    pid_to_iv_and_ojf_gen_response = {d['pid']: d['iv_and_ojf_gen_response'][0] for d in iv_and_ojf_gen_responses}
    
    ig_gen_responses = utils.load_json_line(args.ig_gen_responses_save_path)
    pid_to_ig_gen_response = {d['pid']: d['ig_gen_response'][0] for d in ig_gen_responses}
    
    test_cases_kit_list = []
    for pid in target_pids:
        if pid not in pid_to_iv_and_ojf_gen_response:
            continue
        if pid not in pid_to_ig_gen_response:
            continue
        iv_and_ojf_gen_response = pid_to_iv_and_ojf_gen_response[pid]
        ig_gen_response = pid_to_ig_gen_response[pid]
        test_cases_kit = extract_test_cases_kit(
            pid=pid,
            iv_and_ojf_gen_response=iv_and_ojf_gen_response,
            ig_gen_response=ig_gen_response,
        )
        test_cases_kit_list.append(test_cases_kit)
    utils.save_json_line(test_cases_kit_list, args.test_cases_kit_save_path)
    
if __name__ == '__main__':
    nest_asyncio.apply()
    main()