from typing import Text, List, NamedTuple

from nltk.tokenize import word_tokenize

from mixin import NameMixIn


class Token(NamedTuple):
    name: str
    idx: int


class Tokenizer(NameMixIn):
    """
    Convert text into word tokens using NLTK word_tokenize method.

    Optionally:
        - Converts text to lowercase
        - Append end token to the end of sequence
    """
    def __init__(self, end_token: Token = None, do_lower: bool = True):
        self.end_token = end_token
        self.do_lower = do_lower

    def __repr__(self):
        return f"{self.name}(end_token={self.end_token}, do_lower={self.do_lower})"

    def tokenize(self, text: Text) -> List[Text]:
        """
        Converts text into word tokens.
        """
        if self.do_lower:
            text = text.lower()

        tokens = word_tokenize(text)

        if self.end_token:
            tokens += [self.end_token.name]

        return tokens
