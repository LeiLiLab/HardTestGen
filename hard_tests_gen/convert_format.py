import json
from argparse import ArgumentParser

from tqdm import tqdm

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for entry in data:
            file.write(json.dumps(entry) + '\n')

def convert_format(d):
    """
    Convert the format of a test case dictionary from the old structure to the new structure.
    Old structure: the raw output of HardTestGen/hard_tests_gen/test_cases_generation.py.
    New structure: the format of the dataset on HuggingFace.
    """
    if d['status'] != 'success':
        return None # only convert successful test cases
    test_cases_kit = d['test_cases_kit']
    converted_d = {
        'pid': d['pid'],
        'test_cases_kit': {
            'pid': d['pid'],
            'output_judging_function': test_cases_kit['output_judging_function'],
            'input_validator': test_cases_kit['input_validator'],
            'LLMGen': test_cases_kit['input_generation']['LLMGen_input'],
            'RPGen': None,
            'SPGen': None,
            'HackGen': test_cases_kit['input_generation']['HackGen_input_generator'],
        },
        'mapping': {
            'LLMGen': d['mapping']['LLMGen'],
            'RPGen': [],
            'SPGen': [],
            'HackGen': d['mapping']['HackGen'],
        },
        'test_cases': d['test_cases'],
    }
    rpgen_spgen_code = test_cases_kit['input_generation']['RPGen_SPGen_input_generator']['code']
    if 'gen_range_based_input' in rpgen_spgen_code:
        converted_d['test_cases_kit']['RPGen'] = test_cases_kit['input_generation']['RPGen_SPGen_input_generator']
        converted_d['mapping']['RPGen'] = d['mapping']['RPGen_SPGen']
    elif 'gen_stratified_input' in rpgen_spgen_code:
        converted_d['test_cases_kit']['SPGen'] = test_cases_kit['input_generation']['RPGen_SPGen_input_generator']
        converted_d['mapping']['SPGen'] = d['mapping']['RPGen_SPGen']
    else:
        return None # unrecognized RPGen/SPGen generator
    return converted_d


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSONL file with old format.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output JSONL file with new format.')
    args = parser.parse_args()
    
    print(f'Loading data from {args.input_file}...')
    old_data = load_jsonl(args.input_file)
    
    print('Converting data format...')
    new_data = []
    for d in tqdm(old_data):
        converted_d = convert_format(d)
        if converted_d is not None:
            new_data.append(converted_d)

    print(f'Saving converted data to {args.output_file}...')
    save_jsonl(new_data, args.output_file)
    
if __name__ == '__main__':
    main()