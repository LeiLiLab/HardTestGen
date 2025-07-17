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
    correct_program: str,
    prompt_template: str = 'test_cases_kit_prompt_iv_and_ojf'
):
    template = Template(prompt_templates[prompt_template]['content'])
    prompt = template.render(
        problem_statement=question_content,
        correct_program=correct_program
    )
    return prompt.strip()

def get_ig_gen_prompt(
    question_content: str,
    correct_program: str,
    input_validator: str,
    num_SSI: int = 10,
    prompt_template: str = 'test_cases_kit_prompt_ig',
):
    template = Template(prompt_templates[prompt_template]['content'])
    prompt = template.render(
        num_SSI=str(num_SSI),
        problem_statement=question_content,
        correct_program=correct_program,
        input_validator=input_validator
    )
    return prompt.strip()

def parse_test_cases_kit_response(response: str) -> Dict[str, Any]:
    try:
        match = re.search(r'(?<=# Result)[\s\S]*', response)
        result_content = match.group().strip()
        json_str = utils.extract_code(text=result_content, language='json', allow_no_language_label=False, verbose=False)
        result_dict = json.loads(json_str)
        return result_dict
    except:
        return dict()

def get_test_cases_kit_info_msg(test_cases_kit) -> str:
    num_ri_func_names = len(test_cases_kit['input_generation']['regular_input_generator']['func_names'])
    num_hi_func_names = len(test_cases_kit['input_generation']['hacking_input_generator']['func_names'])
    num_ssi = len(test_cases_kit['input_generation']['small_scale_input'])
    info = [
        f"SSI ({num_ssi})"
        f"RI ({num_ri_func_names})"
        f"HI ({num_hi_func_names})"
        f"IV {'(√)' if test_cases_kit['input_validator'] is not None else '(x)'}",
        f"OJF {'(√)' if test_cases_kit['output_judging_function'] is not None else '(x)'}"
    ]
    return ', '.join(info)


def extract_test_cases_kit(pid: str, iv_and_ojf_gen_response: str, ig_gen_response: str) -> Dict[str, Any]:    
    iv_and_ojf_gen_dict = parse_test_cases_kit_response(iv_and_ojf_gen_response)
    input_validator = iv_and_ojf_gen_dict.get('input_validator', None)
    output_judging_function = iv_and_ojf_gen_dict.get('output_judging_function', None)
    
    ig_gen_dict = parse_test_cases_kit_response(ig_gen_response)
    ri_code = ig_gen_dict.get('regular_input_generator', None)
    ri_func_names = get_function_names(ri_code)
    ri_func_names = [name for name in ri_func_names if name.startswith('gen_regular_input')]
    hi_code = ig_gen_dict.get('hacking_input_generator', None)
    hi_func_names = get_function_names(hi_code)
    hi_func_names = [name for name in hi_func_names if name.startswith('gen_hacking_input')]
    ssi_list = ig_gen_dict.get('small_scale_input', [])
    ssi_list = [str(d) for d in ssi_list]
    
    test_cases_kit = {
        'pid': pid,
        'input_generation': {
            'small_scale_input': ssi_list,
            'regular_input_generator': {
                'code': ri_code,
                'func_names': ri_func_names,
            },
            'hacking_input_generator': {
                'code': hi_code,
                'func_names': hi_func_names,
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
    parser.add_argument('--dataset_name', type=str, help='Name of HuggingFace dataset.')
    parser.add_argument('--target_pids_path', type=str, default=None, help='Path to the JSON file containing target problem IDs. If not provided, all problems in the dataset will be used.')
    parser.add_argument('--iv_and_ojf_gen_prompt_template', type=str, default='test_cases_kit_prompt_iv_and_ojf', help='Prompt template for Input Validator (IV) and Output Judging Function (OJF) generation.')
    parser.add_argument('--ig_gen_prompt_template', type=str, default='test_cases_kit_prompt_ig', help='Prompt template for Input Generation (IG) generation')
    parser.add_argument('--num_SSI', type=int, default=10, help='Number of small-scale inputs to generate.')
    parser.add_argument('--iv_and_ojf_gen_responses_save_path', type=str, default='./iv_and_ojf_gen_responses.jsonl', help='Path to save Input Validator (IV) and Output Judging Function (OJF) generation responses.')
    parser.add_argument('--ig_gen_responses_save_path', type=str, default='./ig_gen_responses.jsonl', help='Path to save Input Generation (IG) generation responses.')
    parser.add_argument('--test_cases_kit_save_path', type=str, default='./test_cases_kits.jsonl', help='Path to save the generated test cases kits')
    parser.add_argument('--model_name', type=str, default='gpt-4o', help='Name of LLM to use for generation.')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for the LLM.')
    parser.add_argument('--max_tokens', type=int, default=int(1024 * 2.5), help='Maximum number of tokens for the LLM.')
    parser.add_argument('--num_parallel', type=int, default=100, help='Number of parallel requests to the LLM.')
    args = parser.parse_args()
    logger.info(f"Arguments: {pformat(vars(args))}")
    return args

def main():
    args = parse_args()
    
    pdata_list = datasets.load_dataset(args.dataset_name)['train'].to_list()
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
    num_problems_without_correct_program = 0
    
    for pid in tqdm(target_pids):
        pdata = pid_to_pdata[pid]
        question_content = pdata['question_content']
        correct_programs = utils.get_most_reliable_solutions(
            pdata=pdata,
            language_list=['cpp', 'python3', 'python'],
            lowest_reliability_level=1,
            max_num=1
        )
        if len(correct_programs) == 0:
            num_problems_without_correct_program += 1
            continue
        else:
            correct_program = correct_programs[0]['code']
        iv_and_ojf_gen_prompt = get_iv_and_ojf_gen_prompt(
            question_content=question_content,
            correct_program=correct_program,
            prompt_template=args.iv_and_ojf_gen_prompt_template
        )
        iv_and_ojf_gen_prompt_data.append({
            'pid': pid,
            'iv_and_ojf_gen_prompt': iv_and_ojf_gen_prompt,
        })
    logger.info(f"Num problems without correct program: {num_problems_without_correct_program}")
    
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
    num_problems_without_correct_program = 0
    num_problems_without_input_validator = 0

    ig_gen_prompt_data = []
    for pid in tqdm(target_pids):
        pdata = pid_to_pdata[pid]
        question_content = pdata['question_content']
        correct_programs = utils.get_most_reliable_solutions(
            pdata=pdata,
            language_list=['cpp', 'python3', 'python'],
            lowest_reliability_level=1,
            max_num=1
        )
        if len(correct_programs) == 0:
            num_problems_without_correct_program += 1
            continue
        else:
            correct_program = correct_programs[0]['code']
        if pid not in pid_to_iv:
            num_problems_without_input_validator += 1
            continue
        else:
            input_validator = pid_to_iv[pid]
        ig_gen_prompt = get_ig_gen_prompt(
            question_content=question_content,
            correct_program=correct_program,
            input_validator=input_validator,
            num_SSI=args.num_SSI,
            prompt_template= args.ig_gen_prompt_template
        )
        ig_gen_prompt_data.append({
            'pid': pid,
            'ig_gen_prompt': ig_gen_prompt,
        })
    logger.info(f"Num problems without correct program: {num_problems_without_correct_program}")
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