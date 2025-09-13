import os

import tiktoken


def get_project_base_directory(*args):

    PROJECT_BASE = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            os.pardir,
        )
    )

    if args:
        return os.path.join(PROJECT_BASE, *args)
    return PROJECT_BASE

tiktoken_cache_dir = get_project_base_directory()
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
encoder = tiktoken.get_encoding("cl100k_base")




def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    try:
        return len(encoder.encode(string))
    except Exception:
        return 0


if __name__ == '__main__':
    root = get_project_base_directory()
    print(root)
    print(num_tokens_from_string("hello world"))