import gzip
import json
import os
from typing import Dict, Iterable

import tempdir
import wget
from appdirs import user_cache_dir

CACHE_DIR = user_cache_dir("evalplus")
benchmark_files = {
    "single_line": "https://github.com/openai/human-eval-infilling/raw/master/data/HumanEval-SingleLineInfilling.jsonl.gz",
    "multi_line": "https://github.com/openai/human-eval-infilling/raw/master/data/HumanEval-MultiLineInfilling.jsonl.gz",
    "random_span": "https://github.com/openai/human-eval-infilling/raw/master/data/HumanEval-RandomSpanInfilling.jsonl.gz",
    "random_span_light": "https://github.com/openai/human-eval-infilling/raw/master/data/HumanEval-RandomSpanInfillingLight.jsonl.gz",
    "test": "https://github.com/openai/human-eval-infilling/raw/master/data/example_problem.jsonl",
}


def make_cache(gzip_url, cache_path):
    # Check if human eval file exists in CACHE_DIR
    if not os.path.exists(cache_path):
        # Install HumanEval dataset and parse as jsonl
        print(f"Downloading dataset from {gzip_url}")
        with tempdir.TempDir() as tmpdir:
            plus_gz_path = os.path.join(tmpdir, "data.jsonl.gz")
            wget.download(gzip_url, plus_gz_path)

            with gzip.open(plus_gz_path, "rb") as f:
                plus = f.read().decode("utf-8")

        # create CACHE_DIR if not exists
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)

        # Write the original human eval file to CACHE_DIR
        with open(cache_path, "w") as f:
            f.write(plus)


def read_problems(benchmark_name: str) -> Dict[str, Dict]:
    benchmark_file_names = {
        "single_line": "HumanEval-SingleLineInfilling.jsonl",
        "multi_line": "HumanEval-MultiLineInfilling.jsonl",
        "random_span": "HumanEval-RandomSpanInfilling.jsonl",
        "random_span_light": "HumanEval-RandomSpanInfillingLight.jsonl",
        "test": "example_problem.jsonl",
    }
    benchmark_file_path = os.path.join(CACHE_DIR, benchmark_file_names[benchmark_name])
    make_cache(benchmark_files[benchmark_name], benchmark_file_path)
    return {task["task_id"]: task for task in stream_jsonl(benchmark_file_path)}


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, "rt") as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = "ab"
    else:
        mode = "wb"
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode="wb") as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode("utf-8"))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode("utf-8"))
