from nltk.tokenize import word_tokenize


class Tokenizer:
    def __init__(self, do_lower=True, end_token=None):
        self.do_lower = do_lower
        self.end_token = end_token

    def __repr__(self):
        return "{}: do_lower={}".format(self.__class__.__name__, self.do_lower)

    def __call__(self, text):
        if self.do_lower:
            text = text.lower()

        tokens = word_tokenize(text)

        if self.end_token:
            tokens += [self.end_token]

        return tokens
