# HardTests: Synthesizing High-Quality Test Cases for LLM Coding

🌐 [Project Page](https://leililab.github.io/HardTests/)

## 📂 Dataset

The dataset contains multiple `.jsonl` files named like `test_cases_0.jsonl`, `test_cases_1.jsonl`, etc. Each line of each file contains structured test case objects for one problem.

### 🔗 Download

All files can be downloaded from the following sources:
📎 [Huggingface - Problems](https://huggingface.co/datasets/sigcp/hardtests_problems)
📎 [Huggingface - Test Cases](https://huggingface.co/datasets/sigcp/hardtests_tests)
📎 [Google Drive - Test Cases](https://drive.google.com/drive/folders/1GB2FdSSZGIicNVIwh_SGQfcbp8dU-llh)

Please refer to the Hugging Face dataset pages for a detailed description of the data structure.

## 📦 Environment Setup

### Step 1. Install Python packages

```bash
pip install -r requirements.txt
```

### Step 2. Install Bubblewrap (bwrap)

[Bubblewrap](https://github.com/containers/bubblewrap) is an open-source sandbox tool that allows you to create and manage sandbox environments on Linux systems without requiring root privileges.

**Install with root:**

```bash
sudo apt-get update
sudo apt install bubblewrap
```

**Install without root (build from source):**

```bash
pip install meson ninja
export PATH="$HOME/.local/bin:$PATH"

git clone https://github.com/containers/bubblewrap.git
cd bubblewrap
meson setup build --prefix=$HOME/.local
meson compile -C build
meson install -C build

~/.local/bin/bwrap --version # check installation
bwrap --version # check installation
```

## 📝 How to Run

### Step 1: Generate Test Cases Kit

Run `test_cases_kit_generation.py` to generate the test cases kit for each problem. This step uses LLMs to synthesize input validators (IV), output judging functions (OJF), and input generators (IG) for each problem.

**Example command:**
```bash
python hard_tests_gen/test_cases_kit_generation.py \
  --dataset_name <huggingface_dataset_name> \
  --target_pids_path <target_pids.json> \
  --iv_and_ojf_gen_prompt_template test_cases_kit_prompt_iv_and_ojf \
  --ig_gen_prompt_template test_cases_kit_prompt_ig \
  --num_SSI 10 \
  --iv_and_ojf_gen_responses_save_path ./iv_and_ojf_gen_responses.jsonl \
  --ig_gen_responses_save_path ./ig_gen_responses.jsonl \
  --test_cases_kit_save_path ./test_cases_kits.jsonl \
  --model_name gpt-4o \
  --temperature 0.1 \
  --max_tokens 2560 \
  --num_parallel 10
```

**Arguments:**
- `--dataset_name`: Name of the HuggingFace dataset to use (e.g., `sigcp/hardtests_problems`)
- `--target_pids_path`: (Optional) Path to a JSON file containing a list of target problem IDs. If not provided, all problems in the dataset will be used.
- `--iv_and_ojf_gen_prompt_template`: Prompt template for generating Input Validator and Output Judging Function (default: `test_cases_kit_prompt_iv_and_ojf`).
- `--ig_gen_prompt_template`: Prompt template for Input Generation (default: `test_cases_kit_prompt_ig`).
- `--num_SSI`: Number of small-scale inputs to generate for each problem (default: 10).
- `--iv_and_ojf_gen_responses_save_path`: Path to save LLM responses for IV/OJF generation.
- `--ig_gen_responses_save_path`: Path to save LLM responses for IG generation.
- `--test_cases_kit_save_path`: Path to save the generated test cases kits.
- `--model_name`: LLM model name (e.g., `gpt-4o`, `deepseek`).
- `--temperature`: Sampling temperature for LLM (default: 0.1).
- `--max_tokens`: Maximum tokens for LLM output (default: 2560).
- `--num_parallel`: Number of parallel LLM requests (default: 10).

### Step 2: Generate Test Cases

Run `test_cases_generation.py` to generate concrete test cases (input-output pairs) for each problem, based on the previously generated test cases kit.

```bash
python hard_tests_gen/test_cases_generation.py \
  --problem_data_path <problem_data_path> \
  --test_cases_kit_path ./test_cases_kits.jsonl \
  --test_cases_save_path ./test_cases.jsonl \
  --test_cases_related_contents_save_path ./test_cases_related_contents.jsonl \
  --bwrap_path bwrap \
  --cpp_compiler_path g++ \
  --python_interpreter_path python3 \
  --code_exec_temp_dir ./tmp \
  --max_workers 3 \
  --save_steps 10 \
  --log_steps 10 \
  --start 0 \
  --end 1000000
```

**Arguments:**
- `--problem_data_path`: Path to the problem data file (JSONL format, each line is a problem dict). This can also be a HuggingFace dataset name (e.g., `sigcp/hardtests_problems`).
- `--test_cases_kit_path`: Path to the test cases kit file generated in Step 1.
- `--test_cases_save_path`: Path to save the generated test cases (JSONL format).
- `--test_cases_related_contents_save_path`: Path to save related contents for each test case (JSONL format).
- `--bwrap_path`: Path to the Bubblewrap executable (default: `bwrap`).
- `--cpp_compiler_path`: Path to the C++ compiler (default: `g++`).
- `--python_interpreter_path`: Path to the Python interpreter (default: `python3`).
- `--code_exec_temp_dir`: Directory for temporary code execution files.
- `--max_workers`: Number of parallel workers (default: 3).
- `--save_steps`: Save results every N problems (default: 10).
- `--log_steps`: Log progress every N problems (default: 10).
- `--start`: Start index for problems to process (default: 0).
- `--end`: End index for problems to process (default: 1000000).
- (Other advanced arguments are available, see code for details.)


**Note:**
- You must run Step 1 before Step 2, as Step 2 depends on the test cases kit generated in Step 1.
- Make sure all paths are set correctly and required files exist before running each step.


## Cite Us

Please cite us if you find it useful using the following bibtex.

```latex
@misc{he2025hardtests,
    title={HardTests: Synthesizing High-Quality Test Cases for LLM Coding},
    author={Zhongmou He and Yee Man Choi and Kexun Zhang and Jiabao Ji and Junting Zhou and Dejia Xu and Ivan Bercovich and Aidan Zhang and Lei Li},
    year={2025},
    month={May}
}
```
