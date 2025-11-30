import json
import os
import sys
import time
import yaml
import asyncio
import random
import re
from typing import List, Dict, Any, Tuple, Optional, Union, Set, Callable
import ast
import concurrent.futures
import time
import shutil
import argparse
import traceback
from pprint import pprint, pformat
import gc
from os.path import dirname, abspath
import pathlib
from dataclasses import dataclass, field
import subprocess
from functools import partial
import textwrap
import copy
from pathlib import Path

from tqdm import tqdm
import datasets
import litellm
from loguru import logger
import openai
import nest_asyncio
from jinja2 import Template
from dynaconf import Dynaconf
from func_timeout import func_timeout, FunctionTimedOut
import transformers
import pandas as pd
from codebubble.utils import ExecutionStatus

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from hard_tests_gen import utils
from hard_tests_gen.utils import flatten_data_dict, unflatten_data_dict, remove_none_from_test_cases, get_object_size_mb
from hard_tests_gen.llm_api import get_completion_for_prompts
from hard_tests_gen.local_judge import IntegratedExecutor, run_code_safe, judge_outputs

@dataclass
class InputGenerationArgs:
    num_RPGen_SPGen_input: int = 20
    num_RPGen_SPGen_input_attempts: int = 40
    num_RPGen_SPGen_input_mcop_per_func: int = 10
    num_RPGen_SPGen_input_mcop_per_func_attempts: int = 20
    
    num_HackGen_input_per_func: int = 10
    num_HackGen_input_per_func_attempts: int = 20
    
    input_gen_time_limit_per_attempt: int = 5 # in seconds
    input_gen_time_limit_per_func: int = 40 # in seconds
    input_gen_LLMGen_input_validation_time_limit: int = 10 # in seconds
    input_gen_overall_time_limit: int = 60 * 3 # in seconds
    input_gen_memory_limit: int = 1024 * 15 * 1024 # in KB
    input_gen_apply_validator: bool = True
    
@dataclass
class OutputGenerationArgs:
    output_gen_overall_time_limit: int = 60 * 4 # in seconds
    
    output_gen_time_limit_per_solution_exec: int = 50 # in seconds
    output_gen_solution_exec_memory_limit: int = 1024 * 15 * 1024 # in KB
    output_gen_time_limit_per_input_exec: int = 5 # in seconds
    
    output_gen_time_limit_per_solution_judge: int = 15 # in seconds
    output_gen_solution_judge_memory_limit: int = 1024 * 15 * 1024 # in KB
    output_gen_time_limit_per_output_judge: int = 5 # in seconds
    
    max_code_solutions_to_try: int = 5
    do_verify: bool = True
    output_inconsistent_tolerance: float = 0.0 # when 0 < output_inconsistent_tolerance < 1: means rate, the bigger, the easier to be kept
    min_num_agreements: int = 2 # the minimum number of agreements of outputs to consider it valid.

    test_cases_max_size_per_problem: int = 500 * 1024 # in KB

@dataclass
class OtherArgs:
    problem_data_path: str
    test_cases_kit_path: str
    test_cases_save_path: str
    test_cases_related_contents_save_path: str
    bwrap_path: str
    cpp_compiler_path: str
    python_interpreter_path: str
    code_exec_temp_dir: str
    cpp_compiler_flags: str = '--std=c++20'
    python_args: str = ''
    code_exec_max_input_size: int = 100 * 1024 # in KB
    code_exec_max_output_size: int = 100 * 1024 # in KB
    pid_subset_path: str = None
    max_workers: int = 3
    save_steps: int = 10
    log_steps: int = 10
    start: int = 0
    end: int = 1000000
    multithread: bool = False
    multiprocess: bool = False
    do_shuffle: bool = False
    task_name: str = 'untitled'
    

def generate_inputs(
    gen_input_func_code: str,
    gen_input_func_name: str,
    integrated_executor: IntegratedExecutor,
    workspace: str,
    input_validator_func_code: str = None,
    apply_validator: bool = False,
    num_attempts: int = 10,
    num_input: int = 10,
    time_limit_per_attempt: int = 3,
    overall_time_limit: int = 30,
    memory_limit: int = 4096,
    deduplicate: bool = False,
    pid: str = 'Unknown PID',
) -> List[str]:
    start_overall = time.time()
    if gen_input_func_code is None:
        return []
    if input_validator_func_code is None:
        logger.debug(f"[{pid}] IV code is None.")
        input_validator_func_code = """
def validate_input(input_str: str) -> bool:
    return True
        """.strip()
        apply_validator = False
    
    apply_validator_str = 'True' if apply_validator else 'False'
    code = f"""
{gen_input_func_code}

{input_validator_func_code}

failed_reason = "Unknown error"
try:
    input_str = {gen_input_func_name}()
    assert isinstance(input_str, str)
    if {apply_validator_str}:
        is_valid = validate_input(input_str)
        assert is_valid, 'Generated input is invalid'
except Exception as e:
    failed_reason = str(e)
    input_str = None

if input_str is None:
    print('Failed: ' + failed_reason, end='')
else:
    print('Result: ' + input_str, end='')
""".strip()
    
    input_list = []
    for i in range(num_attempts):
        if time.time() - start_overall > overall_time_limit:
            logger.debug(f'[{pid}] Overall time limit exceeded when generating inputs.')
            # no need to fill the rest of the input_list with None, just break
            break
        try:
            exec_results = run_code_safe(
                code=code,
                language='python3',
                input_list=[''],
                integrated_executor=integrated_executor,
                workspace=workspace,
                time_limit=time_limit_per_attempt,
                overall_time_limit=time_limit_per_attempt,
                memory_limit=memory_limit,
                code_id=f'{pid}_generate_inputs_{i}',
            )
            if len(exec_results) != 1:
                logger.error(f'[{pid}] Unexpected number of results: {len(exec_results)}.')
                # logger.info('TEMP_STATS: Failed 1')
                continue
            exec_result = exec_results[0]
            if exec_result.status != ExecutionStatus.SUCCESS:
                try:
                    stderr_str = str(exec_result.stderr)
                except:
                    stderr_str = ''
                logger.debug(f'[{pid}] Input generation failed: exec_result.status="{exec_result.status}". exec_result.stderr_str="{stderr_str[:1000]}"')
                # logger.info('TEMP_STATS: Failed 2')
                continue
            try:
                stdout_str = str(exec_result.stdout)
            except:
                stdout_str = ""
            if not stdout_str.startswith('Result: '):
                logger.debug(f'[{pid}] Input generation failed: exec_result.stdout="{stdout_str[:1000]}".')
                # logger.info('TEMP_STATS: Failed 3')
                continue
            input_str = stdout_str[len('Result: '):]
            # logger.info('TEMP_STATS: Success')
            input_list.append(input_str)
            if len(input_list) >= num_input:
                break
        except Exception as e:
            logger.debug(f'[{pid}] Input generation failed: e="{str(e)}".')
            continue
            
    if deduplicate:
        input_list = list(set(input_list))
    return input_list
    
def validate_inputs_and_filter(
    inputs: List[str],
    input_validator_func_code: Optional[str],
    workspace: str,
    integrated_executor: IntegratedExecutor,
    overall_time_limit: int = 10,
    memory_limit: int = 4096,
    pid: str = 'unknown PID',
    deduplicate: bool = False
) -> List[str]:
    if input_validator_func_code is None:
        logger.warning(f'[{pid}] input_validator_func_code is None. Skipping input validation.')
        return inputs
    
    code = f"""
import sys
import json

{input_validator_func_code}

json_str = sys.stdin.read()
input_str_list = json.loads(json_str)

filtered_input_list = []
for input_str in input_str_list:
    is_valid = False
    try:
        is_valid = validate_input(input_str)
    except:
        continue
    if is_valid:
        filtered_input_list.append(input_str)
print('Result: ' + json.dumps(filtered_input_list), end='')
""".strip()    
    try:
        exec_results = run_code_safe(
            code=code,
            language='python3',
            input_list=[json.dumps(inputs)],
            integrated_executor=integrated_executor,
            workspace=workspace,
            time_limit=overall_time_limit,
            overall_time_limit=overall_time_limit,
            memory_limit=memory_limit,
            code_id=f'{pid}_validate_inputs_and_filter',
        )
        if len(exec_results) != 1:
            logger.error(f'[{pid}] Unexpected number of results: {len(exec_results)}.')
            num_invalid = len(inputs)
            logger.info("iubviacyanadav_1 " * num_invalid)
            return []
        exec_result = exec_results[0]
        if exec_result.status != ExecutionStatus.SUCCESS:
            logger.debug(f'[{pid}] Input validation failed: {exec_result.status}.')
            num_invalid = len(inputs)
            logger.info("iubviacyanadav_2 " * num_invalid)
            return []
        stdout_str = exec_result.stdout
        if not stdout_str.startswith('Result: '):
            logger.debug(f'[{pid}] Input validation failed: {str(stdout_str)[:1000]}.')
            num_invalid = len(inputs)
            logger.info("iubviacyanadav_3 " * num_invalid)
            return []
        filtered_inputs = json.loads(stdout_str[len('Result: '):])
        num_invalid = len(inputs) - len(filtered_inputs)
        logger.info("iubviacyanadav_4 " * num_invalid)
        if deduplicate:
            filtered_inputs = list(set(filtered_inputs))
        return filtered_inputs
    except Exception as e:
        logger.debug(f'[{pid}] Input validation failed: {str(e)}.')
        num_invalid = len(inputs)
        logger.info("iubviacyanadav_5 " * num_invalid)
        return []
    
def generate_all_inputs(
    test_cases_kit,
    workspace: str,
    integrated_executor: IntegratedExecutor,
    input_gen_args: Dict[str, Any] = None,
    pid: str = 'unknown PID',
) -> Tuple[Dict[str, Any], List[str]]:
    iv_func_str = test_cases_kit['input_validator']
    
    apply_validator = input_gen_args['input_gen_apply_validator']
    overall_time_limit = input_gen_args['input_gen_overall_time_limit']
    time_limit_per_attempt = input_gen_args['input_gen_time_limit_per_attempt']
    time_limit_per_func = input_gen_args['input_gen_time_limit_per_func']
    LLMGen_input_validation_time_limit = input_gen_args['input_gen_LLMGen_input_validation_time_limit']
    memory_limit = input_gen_args['input_gen_memory_limit']
    
    start_overall = time.time()
    
    # LLMGen Input
    LLMGen_input_list = test_cases_kit['input_generation']['LLMGen_input']
    LLMGen_input_list = validate_inputs_and_filter(
        inputs=LLMGen_input_list,
        input_validator_func_code=iv_func_str,
        workspace=workspace,
        integrated_executor=integrated_executor,
        overall_time_limit=LLMGen_input_validation_time_limit,
        memory_limit=memory_limit,
        pid=pid,
        deduplicate=True,
    )
    
    # RPGen and SPGen Input
    RPGen_SPGen_input_generator = test_cases_kit['input_generation']['RPGen_SPGen_input_generator']
    is_mcop = len(RPGen_SPGen_input_generator['func_names']) >= 2
    code = RPGen_SPGen_input_generator['code']
    num_input = input_gen_args['num_RPGen_SPGen_input_mcop_per_func'] if is_mcop else input_gen_args['num_RPGen_SPGen_input']
    num_attempts = input_gen_args['num_RPGen_SPGen_input_mcop_per_func_attempts'] if is_mcop else input_gen_args['num_RPGen_SPGen_input_attempts']
    RPGen_SPGen_input_list_list = []
    for func_name in RPGen_SPGen_input_generator['func_names']:
        if time.time() - start_overall > overall_time_limit:
            logger.debug(f'[{pid}] Overall time limit exceeded when generating RPGen_SPGen inputs')
            RPGen_SPGen_input_list_list.extend([[] for _ in range(len(RPGen_SPGen_input_generator['func_names']))])
            break
        RPGen_SPGen_input_list = generate_inputs(
            gen_input_func_code=code,
            gen_input_func_name=func_name,
            integrated_executor=integrated_executor,
            workspace=workspace,
            input_validator_func_code=iv_func_str,
            apply_validator=apply_validator,
            num_input=num_input,
            num_attempts=num_attempts,
            time_limit_per_attempt=time_limit_per_attempt,
            overall_time_limit=time_limit_per_func,
            memory_limit=memory_limit,
            deduplicate=True,
            pid=pid,
        )
        RPGen_SPGen_input_list_list.append(RPGen_SPGen_input_list)
    assert len(RPGen_SPGen_input_list_list) == len(RPGen_SPGen_input_generator['func_names']), f"len(RPGen_SPGen_input_list_list)={len(RPGen_SPGen_input_list_list)}, len(RPGen_SPGen_input_generator['func_names'])={len(RPGen_SPGen_input_generator['func_names'])}."
    
    # HackGen Input
    HackGen_input_generator = test_cases_kit['input_generation']['HackGen_input_generator']
    code = HackGen_input_generator['code']
    num_input = input_gen_args['num_HackGen_input_per_func']
    num_attempts = input_gen_args['num_HackGen_input_per_func_attempts']
    HackGen_input_list_list = []
    for func_name in HackGen_input_generator['func_names']:
        if time.time() - start_overall > overall_time_limit:
            logger.debug(f'[{pid}] Overall time limit exceeded when generating HackGen inputs')
            HackGen_input_list_list.extend([[] for _ in range(len(HackGen_input_generator['func_names']))])
            break
        HackGen_input_list = generate_inputs(
            gen_input_func_code=code,
            gen_input_func_name=func_name,
            integrated_executor=integrated_executor,
            workspace=workspace,
            input_validator_func_code=iv_func_str,
            apply_validator=apply_validator,
            num_input=num_input,
            num_attempts=num_attempts,
            time_limit_per_attempt=time_limit_per_attempt,
            overall_time_limit=time_limit_per_func,
            memory_limit=memory_limit,
            deduplicate=True,
            pid=pid,
        )
        HackGen_input_list_list.append(HackGen_input_list)
    assert len(HackGen_input_list_list) == len(HackGen_input_generator['func_names']), f"len(HackGen_input_list_list)={len(HackGen_input_list_list)}, len(HackGen_input_generator['func_names'])={len(HackGen_input_generator['func_names'])}."

    input_dict = {
        'LLMGen': LLMGen_input_list,
        'RPGen_SPGen': RPGen_SPGen_input_list_list,
        'HackGen': HackGen_input_list_list,
    }
    mapping, input_list = flatten_data_dict(input_dict)
    return mapping, input_list

def generate_outputs(
    code_solution: str,
    code_solution_language: str,
    inputs: List[str],
    workspace: str,
    integrated_executor: IntegratedExecutor,
    overall_time_limit: int = 100,
    time_limit_per_input: int = 5,
    memory_limit: int = 4096 * 1024,
    sid: str = 'unknown SID',
) -> Optional[List[str]]:
    if len(inputs) == 0:
        logger.debug(f"[{sid}] No inputs to generate outputs.")
        return []
    
    if code_solution_language not in ['cpp', 'python3', 'python']:
        logger.debug(f"[{sid}] Unsupported language: {code_solution_language}.")
        return None
    
    exec_results = run_code_safe(
        code=code_solution,
        language=code_solution_language,
        input_list=inputs,
        integrated_executor=integrated_executor,
        workspace=workspace,
        time_limit=time_limit_per_input,
        overall_time_limit=overall_time_limit,
        memory_limit=memory_limit,
        code_id=sid,
    )
    if len(exec_results) != len(inputs):
        logger.error(f"[{sid}] Unexpected number of results: {len(exec_results)}. Expected: {len(inputs)}.")
        return None

    if exec_results[0].status == ExecutionStatus.COMPILE_ERROR:
        stderr_str = exec_results[0].stderr
        stderr_str = stderr_str[:1000] if isinstance(stderr_str, str) else ""
        logger.debug(f"[{sid}] Compilation error. stderr: {stderr_str}")
        return None

    generated_outputs = []
    for exec_res in exec_results:
        if exec_res.status != ExecutionStatus.SUCCESS:
            try:
                stderr_str = str(exec_res.stderr)
            except:
                stderr_str = ""
            logger.debug(f'[{sid}] Execution failed. exec_res.status="{exec_res.status}. exec_res.stderr="{stderr_str[:1000]}"')
            generated_outputs.append(None)
        else:
            stdout_str = exec_res.stdout
            if not isinstance(stdout_str, str):
                logger.debug(f"[{sid}] Execution failed. Output is not a string.")
                generated_outputs.append(None)
            else:
                generated_outputs.append(stdout_str)

    generated_outputs_info = [1 if d is not None else 0 for d in generated_outputs]
    logger.debug(f"[{sid}] Generated {sum(generated_outputs_info)} valid outputs for {len(inputs)} inputs. Results: {generated_outputs_info}")
    if len(generated_outputs) != len(inputs):
        logger.error(f"[{sid}] len(generated_outputs)={len(generated_outputs)}, len(inputs)={len(inputs)}. They should be equal.")
        return None
    if sum(generated_outputs_info) < 0.5 * len(generated_outputs_info):
        logger.debug(f"[{sid}] Less than 50% valid outputs generated. Regard as failed.")
        return None
    return generated_outputs


def generate_outputs_and_verify(
    code_solutions: List[Dict[str, Any]],
    workspace: str,
    integrated_executor: IntegratedExecutor,
    input_list: List[str],
    output_generation_args: Dict[str, Any],
    output_judging_function_code: str,
    pid: str,
) -> Optional[List[str]]:
    max_code_solutions_to_try = output_generation_args['max_code_solutions_to_try']
    do_verify = output_generation_args['do_verify']
    output_inconsistent_tolerance = output_generation_args['output_inconsistent_tolerance']
    min_num_agreements = output_generation_args['min_num_agreements']
    
    overall_time_limit = output_generation_args['output_gen_overall_time_limit']
    
    time_limit_per_solution_exec = output_generation_args['output_gen_time_limit_per_solution_exec']
    solution_exec_memory_limit = output_generation_args['output_gen_solution_exec_memory_limit']
    time_limit_per_input_exec = output_generation_args['output_gen_time_limit_per_input_exec']
    
    time_limit_per_solution_judge = output_generation_args['output_gen_time_limit_per_solution_judge']
    solution_judge_memory_limit = output_generation_args['output_gen_solution_judge_memory_limit']
    time_limit_per_output_judge = output_generation_args['output_gen_time_limit_per_output_judge']
    
    start_overall = time.time()
    code_solutions_to_try = code_solutions[:max_code_solutions_to_try]
    
    if len(code_solutions_to_try) == 0:
        logger.debug(f"[{pid}] No code solutions to try.")
        return {
            'result': None,
            'status': 'output_generation_no_code_solutions'
        }
    if len(code_solutions_to_try) < min_num_agreements:
        logger.debug(f'[{pid}] Adjusting min_num_agreements from {min_num_agreements} to {len(code_solutions_to_try)} due to limited number of code solutions to try.')
        min_num_agreements = len(code_solutions_to_try)
    
    final_output_list = None
    output_list_list = []
    for code_solution in code_solutions_to_try:
        if time.time() - start_overall > overall_time_limit:
            logger.debug(f'[{pid}] Overall time limit exceeded when generating outputs.')
            return {
                'result': None,
                'status': 'output_generation_time_limit_exceeded'
            }
        code_solution_id = code_solution['code_solution_id']
        
        cur_output_list = generate_outputs(
            code_solution=code_solution['code'],
            code_solution_language=code_solution['language'],
            inputs=input_list,
            workspace=workspace,
            integrated_executor=integrated_executor,
            overall_time_limit=time_limit_per_solution_exec,
            time_limit_per_input=time_limit_per_input_exec,
            memory_limit=solution_exec_memory_limit,
            sid=code_solution_id,
        )
        if cur_output_list is None:
            logger.debug(f"[{pid}] Failed to generate outputs using code solution {code_solution_id}.")
            continue
        logger.debug(f"[{pid}] Generated {len(cur_output_list)} outputs using code solution {code_solution_id}.")
        assert len(cur_output_list) == len(input_list), f"len(cur_output_list)={len(cur_output_list)}, len(input_list)={len(input_list)}. They should be equal."
        if not do_verify or min_num_agreements == 1:
            # no need to verify, or only one code solution is needed
            final_output_list = cur_output_list
            break
        #output_list_list.append((code_solution_id, cur_output_list))
        cur_output_list_copy = copy.deepcopy(cur_output_list)
        
        succeed = False
        num_agreements = 1
        for j in range(0, len(output_list_list)):
            if time.time() - start_overall > overall_time_limit:
                logger.debug(f'[{pid}] Overall time limit exceeded when generating outputs.')
                return {
                    'result': None,
                    'status': 'output_generation_time_limit_exceeded',
                }
            judging_results = judge_outputs(
                inputs=input_list,
                candidate_outputs=output_list_list[j][1],
                reference_outputs=cur_output_list,
                integrated_executor=integrated_executor,
                workspace=workspace,
                overall_time_limit=time_limit_per_solution_judge,
                time_limit_per_output=time_limit_per_output_judge,
                memory_limit=solution_judge_memory_limit,
                code_id=f"{pid}_{code_solution_id}_vs_{output_list_list[j][0]}",
                output_judging_function_code=output_judging_function_code,
            )
            judging_results = [1 if r == 1 else 0 for r in judging_results] # make it either 0 or 1
            num_inconsistent = len(judging_results) - sum(judging_results)
            num_tolerance = int(output_inconsistent_tolerance)
            if 0 < output_inconsistent_tolerance and output_inconsistent_tolerance < 1:
                # percentage
                num_tolerance = int(output_inconsistent_tolerance * len(judging_results))
            logger.debug(f"[{pid}] num_tolerance={num_tolerance}, num_inconsistent={num_inconsistent}")
                
            if num_inconsistent == 0:
                num_agreements += 1
                logger.debug(f"[{pid}] Output verification succeeded for output_list {output_list_list[j][0]} and {code_solution_id}. Current num_agreements={num_agreements}/{min_num_agreements}.")
                
            elif num_inconsistent <= num_tolerance:
                # some outputs are inconsistent, but still acceptable
                num_agreements += 1
                logger.debug(f"[{pid}] Output verification succeeded for output_list {output_list_list[j][0]} and {code_solution_id}. But note that {num_inconsistent} outputs are inconsistent. Results: {judging_results}. num_agreements={num_agreements}/{min_num_agreements}.")
                # set inconsistent outputs to None
                for idx in range(len(judging_results)):
                    if judging_results[idx] == 0:
                        cur_output_list[idx] = None
            else:
                logger.debug(f"[{pid}] Output verification failed for output_list {output_list_list[j][0]} and {code_solution_id}. Results: {judging_results}. num_agreements={num_agreements}/{min_num_agreements}.")
            
            if num_agreements >= min_num_agreements:
                logger.debug(f"[{pid}] Output verification succeeded for output_list {code_solution_id}.")
                succeed = True
                final_output_list = cur_output_list
                break
                
        if succeed:
            break
        else:
            # add the original current output list to the list
            output_list_list.append((code_solution_id, cur_output_list_copy))
    if final_output_list is None:
        if len(output_list_list) < min_num_agreements:
            return {
                'result': None,
                'status': 'output_generation_no_enough_valid_code_solutions'
            }
        else:
            return {
                'result': None,
                'status': 'output_generation_verification_failed'
            }
    else:
        return {
            'result': final_output_list,
            'status': 'success'
        }


def get_test_cases_kit_info_msg(test_cases_kit) -> str:
    num_RPGen_SPGen_input_func_names = len(test_cases_kit['input_generation']['RPGen_SPGen_input_generator']['func_names'])
    num_HackGen_input_func_names = len(test_cases_kit['input_generation']['HackGen_input_generator']['func_names'])
    num_LLMGen_input = len(test_cases_kit['input_generation']['LLMGen_input'])
    info = [
        f"LLMGen input ({num_LLMGen_input})",
        f"RPGen/SPGen input ({num_RPGen_SPGen_input_func_names} func)",
        f"HackGen input ({num_HackGen_input_func_names} func)",
        f"Input Validator {'(√)' if test_cases_kit['input_validator'] is not None else '(x)'}",
        f"Output Judging Function {'(√)' if test_cases_kit['output_judging_function'] is not None else '(x)'}",
    ]
    return ', '.join(info)

def get_test_cases_info_msg(mapping, tc_list):
    info = [
        f"LLMGen ({len(mapping['LLMGen'])})",
        f"RPGen/SPGen ({[len(d) for d in mapping['RPGen_SPGen']]})",
        f"HackGen ({[len(d) for d in mapping['HackGen']]})",
        f"Total ({len(tc_list)})",
    ]
    return ', '.join(info)

def generate_test_cases(
    test_cases_kit: Dict[str, Any],
    code_solutions: List[Dict[str, str]],
    workspace: str,
    integrated_executor: IntegratedExecutor,
    input_gen_args: Dict[str, Any] = None,
    output_gen_args: Dict[str, Any] = None,
    pid: str = 'Unknown PID',
) -> Dict[str, Any]:
    logger.debug(f"[{pid}] Test Cases Kit: {get_test_cases_kit_info_msg(test_cases_kit)}")
    
    # Generate input
    mapping, input_list = generate_all_inputs(
        test_cases_kit=test_cases_kit,
        workspace=workspace,
        integrated_executor=integrated_executor,
        input_gen_args=input_gen_args,
        pid=pid,
    )
    logger.debug(f"[{pid}] Inputs generated. {get_test_cases_info_msg(mapping, input_list)}")
    if input_list is None or len(input_list) == 0:
        result_dict = {
            'status': 'input_generation_failed',
            'mapping': None,
            'encoded_test_cases': None,
        }
        status = result_dict['status']
        logger.error(f"[{pid}] Failed to generate test cases. Status: {status}")
        return result_dict
    
    # Generate output
    output_gen_result_dict = generate_outputs_and_verify(
        code_solutions=code_solutions,
        workspace=workspace,
        integrated_executor=integrated_executor,
        input_list=input_list,
        output_generation_args=output_gen_args,
        output_judging_function_code=test_cases_kit['output_judging_function'],
        pid=pid,
    )
    output_list = output_gen_result_dict['result']
    if output_list is None:
        result_dict = {
            'status': output_gen_result_dict['status'],
            'mapping': None,
            'encoded_test_cases': None,
        }
        status = result_dict['status']
        logger.error(f"[{pid}] Failed to generate test cases. Status: {status}")
        return result_dict
    
    # Make test cases and filter out None
    test_case_list = []
    for input_str, output_str in zip(input_list, output_list):
        test_case_list.append({'input': input_str, 'output': output_str})
    assert len(test_case_list) == len(input_list)
    assert len(test_case_list) == len(output_list)
    mapping, test_case_list = remove_none_from_test_cases(mapping, test_case_list)
    logger.debug(f"[{pid}] Test cases generated and filtered. {get_test_cases_info_msg(mapping, test_case_list)}")
    test_cases_size = get_object_size_mb(test_case_list) * 1024 # KB
    if test_cases_size > output_gen_args['test_cases_max_size_per_problem']:
        logger.debug(f"[{pid}] Test cases size exceeds the limit. Size: {test_cases_size} MB. Limit: {output_gen_args['test_cases_max_size_per_problem']} KB.")
        result_dict = {
            'status': 'test_cases_size_exceeded_limit',
            'mapping': None,
            'encoded_test_cases': None,
        }
        status = result_dict['status']
        logger.error(f"[{pid}] Failed to generate test cases. Status: {status}")
        return result_dict
    
    # Encode test cases
    if test_case_list is not None:
        try:
            encoded_test_cases = utils.encode_testcases(testcases=test_case_list)
            assert isinstance(encoded_test_cases, str)
        except Exception as e:
            logger.debug(f'[{pid}] Failed to encode test cases.')
            result_dict = {
                'status': 'test_cases_encoding_failed',
                'mapping': None,
                'encoded_test_cases': None,
            }
            status = result_dict['status']
            logger.error(f"[{pid}] Failed to generate test cases. Status: {status}")
            return result_dict
    logger.debug(f"[{pid}] Test cases encoded. Length: {len(encoded_test_cases)} characters.")
    
    logger.info(f"[{pid}] Successfully generated {len(test_case_list)} test cases. {get_test_cases_info_msg(mapping, test_case_list)}")
    return {
        'status': 'success',
        'mapping': mapping,
        'encoded_test_cases': encoded_test_cases,
    }

def generate_test_cases_from_pdata(
    problem_data: Dict[str, Any],
    test_cases_kit: str,
    input_gen_args: InputGenerationArgs,
    output_gen_args: OutputGenerationArgs,
    other_args: OtherArgs,
):
    # Get code solutions
    pid = problem_data['pid']
    code_solutions = utils.get_most_reliable_solutions(
        pdata=problem_data,
        language_list=['cpp', 'python3', 'python'],
        lowest_reliability_level=1,
        max_num=output_gen_args.max_code_solutions_to_try
    )
    code_solution_id_list = [d['code_solution_id'] for d in code_solutions]
    logger.debug(f'[{pid}] Got {len(code_solutions)} code solutions. They are: {code_solution_id_list}.')


    python_args = str(other_args.python_args)
    if python_args == '':
        python_args = []
    else:
        python_args = python_args.split(' ')

    cpp_compiler_flags = str(other_args.cpp_compiler_flags)
    if cpp_compiler_flags == '':
        cpp_compiler_flags = []
    else:
        cpp_compiler_flags = cpp_compiler_flags.split(' ')
        
    # Make integrated executor
    integrated_executor = IntegratedExecutor(
        time_limit=None,
        overall_time_limit=None,
        memory_limit=None,
        max_input_size=other_args.code_exec_max_input_size,
        max_output_size=other_args.code_exec_max_output_size,
        bwrap_path=other_args.bwrap_path,
        cpp_compiler_path=other_args.cpp_compiler_path,
        cpp_compiler_flags=cpp_compiler_flags,
        python_interpreter_path=other_args.python_interpreter_path,
        python_args=python_args,
    )
    
    random_str = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
    workspace = os.path.join(other_args.code_exec_temp_dir, f'{pid}_{random_str}')
    if os.path.exists(workspace):
        raise ValueError(f'Workspace {workspace} already exists.')
    os.makedirs(workspace, exist_ok=True)
    
    # Generate test cases
    result_dict = generate_test_cases(
        test_cases_kit=test_cases_kit,
        code_solutions=code_solutions,
        workspace=workspace,
        integrated_executor=integrated_executor,
        input_gen_args=vars(input_gen_args),
        output_gen_args=vars(output_gen_args),
        pid=pid,
    )
    return result_dict


def _generate_test_cases_worker(args_tuple):
    """
    Wrapper for multiprocessing; keeps child process work minimal and pickle-friendly.
    """
    pdata, tc_kit, input_gen_args, output_gen_args, other_args = args_tuple
    pid = pdata.get('pid', 'unknown_pid')
    kwargs = {
        'problem_data': pdata,
        'test_cases_kit': tc_kit,
        'input_gen_args': input_gen_args,
        'output_gen_args': output_gen_args,
        'other_args': other_args,
    }
    try:
        result_dict = generate_test_cases_from_pdata(**kwargs)
        status = result_dict['status']
        mapping = result_dict['mapping']
        encoded_test_cases = result_dict['encoded_test_cases']
    except Exception as e:
        logger.error(f'[{pid}] Failed to generate test cases due to unexpected error: {e}')
        status = 'unknown_error'
        mapping = None
        encoded_test_cases = None
    return {
        'pid': pid,
        'test_cases_kit': tc_kit,
        'status': status,
        'mapping': mapping,
        'encoded_test_cases': encoded_test_cases,
    }

    
def parse_args():
    parser = transformers.HfArgumentParser((InputGenerationArgs, OutputGenerationArgs, OtherArgs))
    input_gen_args, output_gen_args, other_args = parser.parse_args_into_dataclasses()
    return input_gen_args, output_gen_args, other_args
    
def main():
    input_gen_args, output_gen_args, other_args = parse_args()
    assert not other_args.multithread, 'Multithreading is not supported yet.'
    
    log_folder = f'tc_gen_{other_args.task_name}_{other_args.start}_{other_args.end}_{time.strftime("%Y%m%d-%H%M%S")}'
    os.makedirs(log_folder, exist_ok=True)

    output_file_name = f'{log_folder}/stdout.log'
    f_stdout = open(output_file_name, "w")
    sys.stdout = f_stdout
    
    output_file_name = f'{log_folder}/stderr.log'
    f_stderr = open(output_file_name, "w")
    sys.stderr = f_stderr
    
    logger.remove()
    logging_file_name = f'{log_folder}/logging.log'
    logger.add(logging_file_name, level="DEBUG")
        
    logger.info('Input Generation Args:\n' + pformat(input_gen_args))
    logger.info('Output Generation Args:\n' + pformat(output_gen_args))
    logger.info('Other Args:\n' + pformat(other_args))
    
    code_exec_temp_dir = other_args.code_exec_temp_dir
    max_workers = other_args.max_workers
    save_steps = other_args.save_steps
    test_cases_save_path = other_args.test_cases_save_path
    test_cases_related_contents_save_path = other_args.test_cases_related_contents_save_path
    test_cases_kit_path = other_args.test_cases_kit_path
    
    if not os.path.exists(os.path.dirname(test_cases_save_path)):
        os.makedirs(os.path.dirname(test_cases_save_path), exist_ok=True)
    if not os.path.exists(code_exec_temp_dir):
        os.makedirs(code_exec_temp_dir, exist_ok=True)
    
    test_cases_kit_list = utils.load_json_line(other_args.test_cases_kit_path)
    pid_to_tc_kit: Dict[str, str] = {d['pid']: d for d in test_cases_kit_list}
    logger.info(f'Loaded {len(pid_to_tc_kit)} test cases kit')
    
    if os.path.exists(other_args.problem_data_path):
        problem_data_path = Path(other_args.problem_data_path)
        if problem_data_path.suffix == '.jsonl':
            problem_data_list = utils.load_json_line(other_args.problem_data_path)
        elif problem_data_path.suffix == '.json':
            problem_data_list = utils.load_json(other_args.problem_data_path)
        else:
            raise ValueError(f'Unsupported problem_data_path file format: {other_args.problem_data_path}')
    else:
        problem_data_list = datasets.load_dataset(other_args.problem_data_path)['train'].to_list()
    pid_to_pdata = {d['pid']: d for d in problem_data_list}
    logger.info(f'Loaded {len(pid_to_pdata)} problem data')
    
    pid_subset = set(pid_to_tc_kit.keys()) & set(pid_to_pdata.keys())
    if other_args.pid_subset_path is not None:
        target_pid_subset = set(utils.load_json(other_args.pid_subset_path))
        logger.info(f'There are {len(target_pid_subset)} target pids.')
        pid_subset = pid_subset & target_pid_subset
    
    pid_list = sorted(list(pid_subset))
    logger.info(f'Found {len(pid_list)} valid problems. PIDs: {pid_list[:10]} (first 10)')
    
    pid_list = [pid for pid in pid_list if pid in pid_to_pdata]
    random.seed(0)
    if other_args.do_shuffle:
        random.shuffle(pid_list)
        logger.info(f"Shuffled PIDs: {pid_list[:10]} (first 10)")
    else:
        logger.info(f"Not shuffling PIDs: {pid_list[:10]} (first 10)")
        
    pid_list = pid_list[other_args.start:other_args.end]
    logger.info(f'Processing {len(pid_list)} problem data, from {other_args.start} to {other_args.end}')
    
    finished_pids = set()
    if os.path.exists(test_cases_related_contents_save_path):
        tc_related_contents = utils.load_json_line(test_cases_related_contents_save_path)
        finished_pids.update([d['pid'] for d in tc_related_contents])
        logger.info(f'Found {len(finished_pids)} finished problems in {test_cases_related_contents_save_path}')
    
    f_args = []
    for pid in pid_list:
        if pid in finished_pids:
            continue
        pdata = pid_to_pdata[pid]
        tc_kit = pid_to_tc_kit[pid]
        f_args.append((pdata, tc_kit, input_gen_args, output_gen_args, other_args))
    logger.info(f'Generated {len(f_args)} sets of arguments for generating test cases')
    
    logger.info('Start generating test cases...')
    
    total_finished = 0
    test_cases_results = []
    test_cases_related_contents_results = []
    
    def handle_result(pid: str, tc_kit: str, status: str, mapping: Any, encoded_test_cases: Any, total_tasks: int):
        nonlocal total_finished, test_cases_results, test_cases_related_contents_results
        total_finished += 1
        if total_finished % other_args.log_steps == 0:
            logger.debug(f"Current progress: {total_finished}/{total_tasks}")

        tc_related_contents_res_dict = {
            'pid': pid,
            'test_cases_kit': tc_kit,
            'mapping': mapping,
            'status': status,
        }
        test_cases_related_contents_results.append(tc_related_contents_res_dict)

        tc_res_dict = {
            'pid': pid,
            'test_cases_kit': tc_kit,
            'mapping': mapping,
            'status': status,
            'test_cases': encoded_test_cases,
        }
        test_cases_results.append(tc_res_dict)

        logger.debug(f'Finished generating test cases for {pid}. Total finished: {total_finished}. len(test_cases_results): {len(test_cases_results)}. len(test_cases_related_contents_results): {len(test_cases_related_contents_results)}')

        if len(test_cases_results) >= save_steps or total_finished == total_tasks:
            logger.debug(f'Saving {len(test_cases_results)} sets of test cases to {test_cases_save_path}, and related contents to {test_cases_related_contents_save_path}')
            utils.save_json_line(test_cases_results, test_cases_save_path, do_append=True)
            utils.save_json_line(test_cases_related_contents_results, test_cases_related_contents_save_path, do_append=True)
            test_cases_results = []
            test_cases_related_contents_results = []

    if other_args.multiprocess:
        logger.info(f'Using multiprocessing with max_workers={max_workers}')
        utils.force_delete_folder(code_exec_temp_dir)
        os.makedirs(code_exec_temp_dir, exist_ok=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_generate_test_cases_worker, arg) for arg in f_args]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(f_args)):
                try:
                    res = future.result()
                    pid = res['pid']
                    tc_kit = res['test_cases_kit']
                    status = res['status']
                    mapping = res['mapping']
                    encoded_test_cases = res['encoded_test_cases']
                except Exception as e:
                    pid = 'unknown_pid'
                    tc_kit = None
                    status = 'unknown_error'
                    mapping = None
                    encoded_test_cases = None
                    logger.error(f'Failed to generate test cases in multiprocessing worker due to unexpected error: {e}')
                handle_result(pid, tc_kit, status, mapping, encoded_test_cases, len(f_args))
        utils.force_delete_folder(code_exec_temp_dir)
        os.makedirs(code_exec_temp_dir, exist_ok=True)
    else:
        for arg_idx, arg in tqdm(enumerate(f_args), total=len(f_args)):
            pid, tc_kit = arg[0]['pid'], arg[1]
            kwargs = {
                'problem_data': arg[0],
                'test_cases_kit': arg[1],
                'input_gen_args': arg[2],
                'output_gen_args': arg[3],
                'other_args': arg[4],
            }
            try:
                result_dict = generate_test_cases_from_pdata(**kwargs)
                status = result_dict['status']
                mapping = result_dict['mapping']
                encoded_test_cases = result_dict['encoded_test_cases']
            except Exception as e:
                status = 'unknown_error'
                mapping = None
                encoded_test_cases = None
                logger.error(f'[{pid}] Failed to generate test cases due to unexpected error: {e}')

            handle_result(pid, tc_kit, status, mapping, encoded_test_cases, len(f_args))

            utils.force_delete_folder(code_exec_temp_dir)
            os.makedirs(code_exec_temp_dir, exist_ok=True)

    f_stdout.close()
    f_stderr.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

if __name__ == '__main__':
    print('Starting test cases generation...')
    main()
