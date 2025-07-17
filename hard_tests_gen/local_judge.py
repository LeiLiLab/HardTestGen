from typing import List, Optional, Callable, Tuple, Dict
import json
import os
import time
import copy
import random
import string
import concurrent.futures
import shutil

from tqdm import tqdm
from loguru import logger

from codebubble.utils import ResourceLimits, ExecutionResult, ExecutionStatus
from codebubble.sandbox.bwrap import BwrapSandbox, BwrapSandboxConfig
from codebubble.executor.python import PythonExecutor, PythonExecutorConfig
from codebubble.executor.cpp import CppExecutor, CppExecutorConfig

class IntegratedExecutor:
    def __init__(
        self,
        
        # resource limits
        time_limit: int = 5,
        overall_time_limit: int = 20,
        memory_limit: int = 2 * 1024 * 1024, # 2GB
        max_input_size: int = 200 * 1024, # 200MB
        max_output_size: int = 200 * 1024, # 200MB
        
        # sandbox config
        bwrap_path: str = 'bwrap',
        
        # C++ config
        cpp_compiler_path: str = 'g++',
        cpp_compiler_flags: Optional[List[str]] = None,
        
        # Python config
        python_interpreter_path: str = 'python',
        python_args: Optional[List[str]] = None,
    ):
        self.time_limit = time_limit
        self.overall_time_limit = overall_time_limit
        self.memory_limit = memory_limit
        self.max_input_size = max_input_size
        self.max_output_size = max_output_size
        self.bwrap_path = bwrap_path
        self.cpp_compiler_path = cpp_compiler_path
        self.cpp_compiler_flags = [] if cpp_compiler_flags is None else cpp_compiler_flags
        self.python_interpreter_path = python_interpreter_path
        self.python_args = [] if python_args is None else python_args
    
    def make_resource_limits(
        self,
        time_limit: Optional[int] = None,
        overall_time_limit: Optional[int] = None,
        memory_limit: Optional[int] = None,
        max_input_size: Optional[int] = None,
        max_output_size: Optional[int] = None,
    ) -> ResourceLimits:
        return ResourceLimits(
            time_limit=self.time_limit if time_limit is None else time_limit,
            overall_time_limit=self.overall_time_limit if overall_time_limit is None else overall_time_limit,
            memory_limit=self.memory_limit if memory_limit is None else memory_limit,
            max_input_size=self.max_input_size if max_input_size is None else max_input_size,
            max_output_size=self.max_output_size if max_output_size is None else max_output_size,
        )
        
    def make_sandbox(self, workspace: str) -> BwrapSandbox:
        config = BwrapSandboxConfig(
            workspace=workspace,
            bwrap_path=self.bwrap_path,
        )
        sandbox = BwrapSandbox(config)
        return sandbox
    
    def make_cpp_executor(self, workspace: str) -> CppExecutor:
        config = CppExecutorConfig(
            compiler_path=self.cpp_compiler_path,
            compiler_flags=self.cpp_compiler_flags,
        )
        sandbox = self.make_sandbox(workspace)
        executor = CppExecutor(config, sandbox)
        return executor
    
    def make_python_executor(self, workspace: str) -> PythonExecutor:
        config = PythonExecutorConfig(
            interpreter_path=self.python_interpreter_path,
            args=self.python_args,
        )
        sandbox = self.make_sandbox(workspace)
        executor = PythonExecutor(config, sandbox)
        return executor
    
def default_output_judging_function(
    input_str: str,
    candidate_output: Optional[str],
    reference_output: Optional[str],
) -> bool:
    if candidate_output is None and reference_output is None:
        return True
    if not isinstance(candidate_output, str) or not isinstance(reference_output, str):
        return False
    normalized_candidate_output = '\n'.join(line.rstrip() for line in candidate_output.rstrip().splitlines())
    normalized_reference_output = '\n'.join(line.rstrip() for line in reference_output.rstrip().splitlines())
    return normalized_candidate_output == normalized_reference_output


def get_output_judging_code(output_judging_function_code: Optional[str]) -> Optional[str]:
    if output_judging_function_code is None:
        return None
    code = f"""
import sys
import json

{output_judging_function_code}

judging_result = None
failed_reason = "Unknown error"
try:
    json_str = sys.stdin.read()
    data_dict = json.loads(json_str)
    
    judging_result = output_judging_function(
        input_str=data_dict['input_str'],
        candidate_output=data_dict['candidate_output'],
        reference_output=data_dict['reference_output']
    )
except Exception as e:
    failed_reason = str(e)
    judging_result = None

if judging_result is None:
    print('Failed: ' + failed_reason, end='')
else:
    judging_result = 'True' if judging_result else 'False'
    print('Result: ' + judging_result, end='')
""".strip()
    return code
    
def parse_output_judging_result(judge_result: ExecutionResult, code_id: str = 'Unknown Code ID') -> Optional[bool]:
    if judge_result.status != ExecutionStatus.SUCCESS:
        logger.debug(f'[{code_id}] Custom output judging function failed: {judge_result.status}.')
        return None
    stdout_str = judge_result.stdout
    if not stdout_str.startswith('Result: '):
        logger.debug(f'[{code_id}] Custom output judging function failed: {str(stdout_str)[:1000]}.')
        return None
    stdout_str = stdout_str[len('Result: '):]
    if stdout_str == 'True':
        return True
    elif stdout_str == 'False':
        return False
    logger.debug(f'[{code_id}] Custom output judging function got unexpected result: {stdout_str}.')
    return None

def get_pass_rate(verdicts: List[int]) -> float:
    if len(verdicts) == 0:
        return 0.0
    pass_count = sum(1 for verdict in verdicts if verdict == 1)
    total_count = len(verdicts)
    pass_rate = pass_count / total_count
    return pass_rate


def run_and_judge_code(
    code: str,
    language: str,
    inputs: List[str],
    outputs: List[str],
    integrated_executor: IntegratedExecutor,
    workspace: str,
    run_time_limit: int = 5,
    run_memory_limit: int = 1024 * 2,
    judge_time_limit: int = 10,
    judge_memory_limit: int = 1024 * 2,
    overall_time_limit: int = 20,
    output_judging_function_code: Optional[str] = None,
    code_id: str = 'Unknown Code ID',
    early_exit: bool = False,
) -> Tuple[str, List[ExecutionResult], List[int]]:
    assert len(inputs) == len(outputs), f'Input and output lists must have the same length. {len(inputs)} != {len(outputs)}'
    
    os.makedirs(workspace, exist_ok=True)
    
    run_workspace = os.path.join(workspace, 'run')
    judge_workspace = os.path.join(workspace, 'judge')
    
    run_code = code
    judge_code = get_output_judging_code(output_judging_function_code)
    
    if language == 'cpp':
        run_executor = integrated_executor.make_cpp_executor(run_workspace)
    elif language in {'python3', 'python'}:
        run_executor = integrated_executor.make_python_executor(run_workspace)
    else:
        raise ValueError(f'Unsupported language: {language}')
    judge_executor = integrated_executor.make_python_executor(judge_workspace)
    
    run_resource_limits = integrated_executor.make_resource_limits(
        overall_time_limit=None,
        time_limit=run_time_limit,
        memory_limit=run_memory_limit,
    )    
    judge_resource_limits = integrated_executor.make_resource_limits(
        overall_time_limit=None,
        time_limit=judge_time_limit,
        memory_limit=judge_memory_limit,
    )
    
    run_executor.sandbox.reset_workspace()
    run_prepare_result = run_executor.prepare(run_workspace, run_code)
    judge_executor.sandbox.reset_workspace()
    if judge_code is not None:
        logger.debug(f'[{code_id}] Using custom output judging function.')
        judge_prepare_result = judge_executor.prepare(judge_workspace, judge_code)
    else:
        judge_prepare_result = None
    
    run_compile_time = run_prepare_result.get('compile_time', None)
    run_compile_return_code = run_prepare_result.get('compile_return_code', None)
    if run_compile_return_code is not None and run_compile_return_code != 0:
        run_compile_stderr = run_prepare_result.get("compile_stderr", '')
        run_compile_stderr = str(run_compile_stderr)[:300]
        logger.debug(f'[{code_id}] Compilation error. Return code: {run_compile_return_code}. Stderr: {run_compile_stderr}')
        run_results = [ExecutionResult(
            status=ExecutionStatus.COMPILE_ERROR,
            compile_time=run_compile_time,
            error_info=f"Compilation failed. Return code: {run_compile_return_code}. Stderr: {run_compile_stderr}",
        ) for _ in inputs]
        verdicts = [-1 for _ in inputs]
        logger.debug(f'[{code_id}] Pass rate: 0.0. Passed list: {verdicts}')
        shutil.rmtree(workspace, ignore_errors=True)
        return code_id, run_results, verdicts
    
    run_results = []
    verdicts = []
    t0 = time.time()
    for tc_idx in range(len(inputs)):
        if time.time() - t0 > overall_time_limit:
            logger.debug(f'[{code_id}] Overall time limit exceeded. Skipping remaining test cases.')
            run_results += [ExecutionResult(status=ExecutionStatus.SKIPPED) for _ in range(len(inputs) - tc_idx)]
            verdicts += [-1 for _ in range(len(inputs) - tc_idx)]
            break
        input_str = inputs[tc_idx]
        reference_output_str = outputs[tc_idx]
        run_result = run_executor.single_run(
            code=run_code,
            input_str=input_str,
            limits=run_resource_limits,
            prepare_result=run_prepare_result,
        )
        if run_result.status != ExecutionStatus.SUCCESS:
            logger.debug(f'[{code_id}] Run error. Status: {run_result.status}.')
            verdict = 0
        else:
            candidate_output_str = run_result.stdout
            if judge_code is None:
                verdict = default_output_judging_function(input_str, candidate_output_str, reference_output_str)
                verdict = 1 if verdict else 0
            else:
                judge_input = {
                    'input_str': input_str,
                    'candidate_output': candidate_output_str,
                    'reference_output': reference_output_str,
                }
                judge_input_str = json.dumps(judge_input)
                judge_result = judge_executor.single_run(
                    code=judge_code,
                    input_str=judge_input_str,
                    limits=judge_resource_limits,
                    prepare_result=judge_prepare_result,
                )
                try:
                    verdict = parse_output_judging_result(judge_result, code_id)
                except Exception as e:
                    logger.debug(f'[{code_id}] Failed to parse output judging result: {str(e)}')
                    verdict = None
                verdict = 1 if verdict else 0
            # if verdict == 1:
            #     logger.debug(f'[{code_id}] Test case #{tc_idx} passed.')
            # else:
            #     logger.debug(f'[{code_id}] Test case #{tc_idx} failed.')
        run_results.append(run_result)
        verdicts.append(verdict)
    
        if early_exit and verdict != 1:
            logger.debug(f'[{code_id}] Early exit. Verdict: {verdict}. Skipping remaining test cases.')
            run_results += [ExecutionResult(status=ExecutionStatus.SKIPPED) for _ in range(len(inputs) - tc_idx - 1)]
            verdicts += [-1 for _ in range(len(inputs) - tc_idx - 1)]
            break
        
    assert len(run_results) == len(inputs)
    assert len(verdicts) == len(inputs)
    
    pass_rate = get_pass_rate(verdicts)
    logger.debug(f'[{code_id}] Pass rate: {pass_rate:.2%}. Passed list: {verdicts}')
    
    shutil.rmtree(workspace, ignore_errors=True)
    return code_id, run_results, verdicts

def judge_outputs(
    inputs: List[str],
    candidate_outputs: List[Optional[str]],
    reference_outputs: List[Optional[str]],
    integrated_executor: IntegratedExecutor,
    workspace: str,
    overall_time_limit: int = 20,
    time_limit_per_output: int = 5,
    memory_limit: int = 1024 * 1024 * 2,
    code_id: str = 'Unknown Code ID',
    output_judging_function_code: Optional[str] = None,
) -> List[int]:
    os.makedirs(workspace, exist_ok=True)
    judge_workspace = os.path.join(workspace, 'judge')
    judge_code = get_output_judging_code(output_judging_function_code)
    judge_executor = integrated_executor.make_python_executor(judge_workspace)
    judge_resource_limits = integrated_executor.make_resource_limits(
        overall_time_limit=None,
        time_limit=time_limit_per_output,
        memory_limit=memory_limit,
    )
    judge_executor.sandbox.reset_workspace()
    if judge_code is not None:
        logger.debug(f'[{code_id}] Using custom output judging function.')
        judge_prepare_result = judge_executor.prepare(judge_workspace, judge_code)
    else:
        judge_prepare_result = None
    
    verdict_list = []
    t0 = time.time()
    for tc_idx in range(len(inputs)):
        if time.time() - t0 > overall_time_limit:
            logger.debug(f'[{code_id}] Overall time limit exceeded. Skipping remaining test cases.')
            verdict_list += [-1 for _ in range(len(inputs) - tc_idx)]
            break
        input_str = inputs[tc_idx]
        candidate_output_str = candidate_outputs[tc_idx]
        reference_output_str = reference_outputs[tc_idx]
        
        if candidate_output_str is None and reference_output_str is None:
            verdict = 1
        else:
            if judge_code is None:
                verdict = default_output_judging_function(input_str, candidate_output_str, reference_output_str)
                verdict = 1 if verdict else 0
            else:
                judge_input = {
                    'input_str': input_str,
                    'candidate_output': candidate_output_str,
                    'reference_output': reference_output_str,
                }
                judge_input_str = json.dumps(judge_input)
                judge_result = judge_executor.single_run(
                    code=judge_code,
                    input_str=judge_input_str,
                    limits=judge_resource_limits,
                    prepare_result=judge_prepare_result,
                )
                try:
                    verdict = parse_output_judging_result(judge_result, code_id)
                except Exception as e:
                    logger.debug(f'[{code_id}] Failed to parse output judging result: {str(e)}')
                    verdict = 0
            verdict = 1 if verdict else 0
        verdict_list.append(verdict)
    
    assert len(verdict_list) == len(inputs)
    return verdict_list
    
def run_and_judge_code_safe(
    code: str,
    language: str,
    inputs: List[str],
    outputs: List[str],
    integrated_executor: IntegratedExecutor,
    workspace: str,
    run_time_limit: int = 5,
    run_memory_limit: int = 1024 * 2,
    judge_time_limit: int = 10,
    judge_memory_limit: int = 1024 * 2,
    overall_time_limit: int = 20,
    output_judging_function_code: Optional[str] = None,
    code_id: str = 'Unknown Code ID',
    early_exit: bool = False,
) -> Tuple[str, List[ExecutionResult], List[int]]:
    try:
        return run_and_judge_code(
            code=code,
            language=language,
            inputs=inputs,
            outputs=outputs,
            integrated_executor=integrated_executor,
            workspace=workspace,
            run_time_limit=run_time_limit,
            run_memory_limit=run_memory_limit,
            judge_time_limit=judge_time_limit,
            judge_memory_limit=judge_memory_limit,
            overall_time_limit=overall_time_limit,
            output_judging_function_code=output_judging_function_code,
            code_id=code_id,
            early_exit=early_exit,
        )
    except Exception as e:
        logger.debug(f'[{code_id}] Exception occurred: {str(e)}')
        run_results = [ExecutionResult(status=ExecutionStatus.ERROR) for _ in range(len(inputs))]
        verdicts = [-1 for _ in range(len(inputs))]
        logger.debug(f'[{code_id}] Pass rate: 0.0')
        shutil.rmtree(workspace, ignore_errors=True)
        return code_id, run_results, verdicts

def run_code(
    code: str,
    language: str,
    input_list: List[str],
    integrated_executor: IntegratedExecutor,
    workspace: str,
    time_limit: int = 5,
    overall_time_limit: int = 20,
    memory_limit: int = 2 * 1024 * 1024,
    code_id: str = 'Unknown Code ID',
) -> List[ExecutionResult]:
    """
    Run a single code snippet in the specified language with given inputs.
    """
    run_workspace = os.path.join(workspace, 'run')
    os.makedirs(run_workspace, exist_ok=True)

    run_code = code
    if language == 'cpp':
        run_executor = integrated_executor.make_cpp_executor(run_workspace)
    elif language in {'python3', 'python'}:
        run_executor = integrated_executor.make_python_executor(run_workspace)
    else:
        raise ValueError(f'Unsupported language: {language}')
    run_resource_limits = integrated_executor.make_resource_limits(
        overall_time_limit=overall_time_limit,
        time_limit=time_limit,
        memory_limit=memory_limit,
    )
    run_results = run_executor.run(
        code=run_code,
        inputs=input_list,
        limits=run_resource_limits,
    )
    shutil.rmtree(workspace, ignore_errors=True)
    status_list = [result.status for result in run_results]
    logger.debug(f'[{code_id}] Run results for code in {language}: {status_list}')

    return run_results

def run_code_safe(
    code: str,
    language: str,
    input_list: List[str],
    integrated_executor: IntegratedExecutor,
    workspace: str,
    time_limit: int = 5,
    overall_time_limit: int = 20,
    memory_limit: int = 2 * 1024 * 1024,
    code_id: str = 'Unknown Code ID',
) -> List[ExecutionResult]:
    """
    Run a single code snippet in the specified language with given inputs, handling exceptions.
    """
    try:
        return run_code(
            code=code,
            language=language,
            input_list=input_list,
            integrated_executor=integrated_executor,
            workspace=workspace,
            time_limit=time_limit,
            overall_time_limit=overall_time_limit,
            memory_limit=memory_limit,
            code_id=code_id,
        )
    except Exception as e:
        logger.debug(f'[{code_id}] Exception occurred: {str(e)}')
        run_results = [ExecutionResult(status=ExecutionStatus.ERROR) for _ in range(len(input_list))]
        shutil.rmtree(workspace, ignore_errors=True)
        return run_results


def run_and_judge_codes_multiprocess(
    codes_list: List[List[str]],
    languages_list: List[List[str]],
    test_cases_list: List[List[Dict[str, str]]],
    integrated_executor: IntegratedExecutor,
    base_workspace: str,
    code_ids_list: Optional[List[List[str]]] = None,
    problem_id_list: Optional[List[str]] = None,
    output_judging_function_code_list: Optional[List[Optional[str]]] = None,
    run_time_limit: int = 5,
    run_memory_limit: int = 1024 * 2,
    judge_time_limit: int = 10,
    judge_memory_limit: int = 1024 * 2,
    overall_time_limit: int = 20,
    early_exit: bool = False,
    max_workers: int = 4,
) -> List[Tuple[str, List[ExecutionResult], List[int]]]:
    if code_ids_list is None:
        code_ids_list = []
        for i in range(len(codes_list)):
            code_ids = []
            for j in range(len(codes_list[i])):
                code_ids.append(f'code_{i}_{j}')
            code_ids_list.append(code_ids)
    if problem_id_list is None:
        problem_id_list = []
        for i in range(len(codes_list)):
            problem_id_list.append(f'problem_{i}')
    if output_judging_function_code_list is None:
        output_judging_function_code_list = [None for _ in range(len(codes_list))]

    param_dict_list = []
    for problem_idx in range(len(codes_list)):
        codes = codes_list[problem_idx]
        languages = languages_list[problem_idx]
        test_cases = test_cases_list[problem_idx]
        code_ids = code_ids_list[problem_idx]
        output_judging_function_code = output_judging_function_code_list[problem_idx]
        inputs = [test_case['input'] for test_case in test_cases]
        outputs = [test_case['output'] for test_case in test_cases]
        
        for code_idx in range(len(codes)):
            code = codes[code_idx]
            code_id = code_ids[code_idx]
            language = languages[code_idx]
            random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
            workspace = os.path.join(base_workspace, code_id + '_' + random_str)
            param_dict = {
                'code': code,
                'language': language,
                'inputs': inputs,
                'outputs': outputs,
                'integrated_executor': integrated_executor,
                'workspace': workspace,
                'run_time_limit': run_time_limit,
                'run_memory_limit': run_memory_limit,
                'judge_time_limit': judge_time_limit,
                'judge_memory_limit': judge_memory_limit,
                'overall_time_limit': overall_time_limit,
                'output_judging_function_code': output_judging_function_code,
                'code_id': code_id,
                'early_exit': early_exit,
            }
            param_dict_list.append(param_dict)
    
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_and_judge_code, **param_dict) for param_dict in param_dict_list]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Evaluating solutions"):
            results.append(future.result())
    
    # Make sure to clean up the workspaces
    for param_dict in param_dict_list:
        workspace = param_dict['workspace']
        if os.path.exists(workspace):
            try:
                shutil.rmtree(workspace)
            except Exception as e:
                continue
    
    code_id_to_result = {result[0]: result for result in results}
    stats_list = []
    for problem_idx in range(len(codes_list)):
        problem_id = problem_id_list[problem_idx]
        code_ids = code_ids_list[problem_idx]
        test_cases = test_cases_list[problem_idx]
        codes = codes_list[problem_idx]
        stats = {
            'problem_id': problem_id,
            'codes': codes,
            'code_ids': code_ids,
            'num_test_cases': len(test_cases),
            'test_cases_passed_list_list': [],
            'test_cases_pass_rate_list': [],
        }
        for code_idx in range(len(codes)):
            code_id = code_ids[code_idx]
            result = code_id_to_result[code_id]
            code_id, run_results, verdicts = result
            stats['test_cases_passed_list_list'].append(verdicts)
            if len(verdicts) != len(test_cases):
                logger.debug(f'[{code_id}] Test case length mismatch. Expected: {len(test_cases)}, Got: {len(verdicts)}')
            pass_rate = get_pass_rate(verdicts)
            stats['test_cases_pass_rate_list'].append(pass_rate)
        stats_list.append(stats)
    return stats_list
