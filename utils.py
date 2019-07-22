import json
from typing import Text, Dict, Any


def read_params(path: Text) -> Dict[Text, Any]:
    with open(path, "r") as f:
        return json.load(f)
