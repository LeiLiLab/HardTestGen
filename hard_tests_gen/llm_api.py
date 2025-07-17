import json
import os
import sys
import time
import yaml
import asyncio
import random
import re
from typing import List, Dict, Any, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
import openai
import litellm
import nest_asyncio
nest_asyncio.apply() # so that we can run asyncio in jupyter notebook

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from hard_tests_gen import utils

litellm_model_alias_map = {
    "deepseek-coder": "deepseek/deepseek-coder", # it's merged into deepseek-chat on 2024-09-09
    'deepseek-chat': 'deepseek/deepseek-chat',
    'deepseek': 'deepseek/deepseek-chat',
    'gpt-4o': 'gpt-4o',
    'gpt-4-turbo': 'gpt-4-turbo',
    'gpt-4o-mini': 'gpt-4o-mini',
    'gpt-3.5-turbo': 'gpt-3.5-turbo',
}

api_key_map = {
    'deepseek': None,
    'openai': None,
}

def set_api_key():
    if 'deepseek' in api_key_map and api_key_map['deepseek'] is not None:
        os.environ['DEEPSEEK_API_KEY'] = api_key_map['deepseek']
        logger.info(f"Deepseek API key set.")
    if 'openai' in api_key_map and api_key_map['openai'] is not None:
        os.environ['OPENAI_API_KEY'] = api_key_map['openai']
        logger.info(f"OpenAI API key set.")

set_api_key()

async def get_deepseek_response(
    messages: List[Dict[str, str]],
    model_name: str = "deepseek-chat",
    n: int = 1,
    num_parallel_for_n: int = 10,
    **kwargs
) -> List[str]:
    if not hasattr(get_deepseek_response, "client"):
        logger.info(f"Creating client for Deepseek.")
        get_deepseek_response.client = openai.AsyncOpenAI(
            api_key=os.environ['DEEPSEEK_API_KEY'],
            base_url="https://api.deepseek.com"
        )
    client = get_deepseek_response.client
    
    async def semaphore_generate_response() -> str:
        async with semaphore:
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
    semaphore = asyncio.Semaphore(num_parallel_for_n)
    tasks = [semaphore_generate_response() for _ in range(n)]
    responses = await asyncio.gather(*tasks)
    return responses


async def get_completion_async(
    prompt: Union[str, Dict[str, str]],
    model_name: str = "deepseek-chat",
    try_times: int = 5,
    retry_wait_time: int = 2,
    use_litellm: bool = True,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 1024 * 4,
    n: int = 1,
    stop: Optional[List[str]] = None,
) -> Optional[List[str]]:
    if isinstance(prompt, str):
        messages = [{"content": prompt, "role": "user"}]
    else:
        messages = prompt
        
    if model_name == 'mock':
        wait_time = random.randint(1, 5)
        await asyncio.sleep(wait_time)
        return [f"Mock response for {prompt}. Waited for {wait_time} seconds." for _ in range(n)]
    
    for i in range(try_times):
        try:
            if use_litellm:
                response = await litellm.acompletion(
                    model=litellm_model_alias_map[model_name] if model_name in litellm_model_alias_map else model_name,
                    messages=messages,
                    temperature=temperature, top_p=top_p,
                    max_tokens=max_tokens, n=n, stop=stop
                )
                text_list = [r['message']['content'] for r in response['choices']]
            else:
                if model_name.startswith('deepseek'):
                    text_list = await get_deepseek_response(messages, model_name,
                                                            n=n, temperature=temperature,
                                                            top_p=top_p, max_tokens=max_tokens,
                                                            stop=stop)
                else:
                    raise ValueError(f"Model name: {model_name} not supported. Try setting use_litellm to True.")
            
            assert text_list is not None
            return text_list
        except Exception as e:
            logger.error(f"Error in getting completion. {e}")
            if i == try_times - 1:
                return None
            else:
                logger.error(f"Retrying... ({i + 1} / {try_times})")
                time.sleep(retry_wait_time)

def get_completion(*args, **kwargs):
    return asyncio.run(get_completion_async(*args, **kwargs))


async def get_completion_for_prompts_async(
    prompt_list: List[str],
    model_name: str = 'fake',
    id_list: Optional[List[str]] = None,
    num_parallel: int = 10,
    save_path: Optional[str] = None, 
    save_steps: int = 10,
    id_field_name: str = 'idx',
    responses_field_name: str = 'responses',
    do_return: bool = False,
    load_finished: bool = True,
    log_steps: int = 10,
    **kwargs
) -> Optional[List[Dict[str, Any]]]:
    """ Get completions for a list of prompts.
    
    Args:
        prompt_list: The list of prompts to get completions for.
        model_name: The model name to use.
        id_list: The list of ids corresponding to the prompts.
        num_parallel: Number of parallel requests.
        save_path: The path to save the responses.
        save_steps: Number of steps to save the responses.
        id_field_name: The field name for the id.
        responses_field_name: The field name for the response.
        do_return: Whether to return the responses.
        **kwargs: Other keyword arguments for the completion function
    """
    if id_list is None:
        id_list = [str(i) for i in range(len(prompt_list))]
    
    
    logger.info(f'Getting completions for {len(prompt_list)} prompts from {model_name}.')
    
    if save_path is not None and os.path.exists(save_path) and load_finished:
        finished_data = utils.load_json_line(save_path)
        finished_ids = set([d[id_field_name] for d in finished_data])
        logger.info(f'{len(finished_ids)} prompts already finished.')
        new_prompt_list, new_id_list = [], []
        for prompt, id in zip(prompt_list, id_list):
            if id not in finished_ids:
                new_prompt_list.append(prompt)
                new_id_list.append(id)
        prompt_list, id_list = new_prompt_list, new_id_list
        

    async def semaphore_get_completion(prompt: str, idx: int):
        async with semaphore:
            response = await get_completion_async(prompt, model_name, **kwargs)
            return response, idx
    semaphore = asyncio.Semaphore(num_parallel)
    tasks = [semaphore_get_completion(prompt, idx) for idx, prompt in enumerate(prompt_list)]
    
    total_finished = 0
    responses, all_responses = [], []
    for task in asyncio.as_completed(tasks):
        response, idx = await task
        id = id_list[idx]
        d = {id_field_name: id, responses_field_name: response}
        responses.append(d)
        if do_return:
            all_responses.append(d)
        total_finished += 1
        if total_finished % log_steps == 0:
            logger.info(f'Completion finished for prompt {id} ({total_finished} / {len(prompt_list)}).')
        if save_path and (len(responses) >= save_steps or total_finished >= len(prompt_list)):
            utils.save_json_line(responses, save_path, do_append=True)
            logger.info(f'{len(responses)} responses saved into {save_path}.')
            responses = []
    logger.info(f'All completions finished.')
    
    if not do_return:
        return None
    
    if save_path is not None:
        all_responses = utils.load_json_line(save_path)
        return all_responses
    return all_responses

def get_completion_for_prompts(*args, **kwargs):
    return asyncio.run(get_completion_for_prompts_async(*args, **kwargs))
