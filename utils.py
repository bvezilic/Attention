import json
from typing import Text, Dict, Any, List


def read_params(path: Text) -> Dict[Text, Any]:
    """Reads JSON file and return dictionary.
    """
    with open(path, "r") as f:
        return json.load(f)


def filter_tokens(l: List[Any]) -> List[Any]:
    """
    Recursively filters special tokens (PAD, EOS, UNK). When PAD or UNK are encountered, they are skipped. However,
    when EOS is encountered iteration stops i.e. no following tokens are recorded.

    Args:
        l (list): List containing any number of nested lists

    Returns:
        tokens: List of filtered tokens with the same shape
    """
    tokens = []
    for el in l:
        if isinstance(el, list):
            tokens_ = filter_tokens(el)
            tokens.append(tokens_)
        else:
            if el == "[EOS]":
                break
            elif el == "[PAD]" or el == "[UNK]":
                continue
            else:
                tokens.append(el)

    return tokens
