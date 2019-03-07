from nltk.tokenize import word_tokenize


class Tokenizer:
    def __init__(self, end_token=None, do_lower=True):
        self.end_token = end_token
        self.do_lower = do_lower

    def __repr__(self):
        return "{}: end_token={}, do_lower={}".format(self.__class__.__name__, self.end_token, self.do_lower)

    def tokenize(self, text):
        if self.do_lower:
            text = text.lower()

        tokens = word_tokenize(text)

        if self.end_token:
            tokens += [self.end_token.name]

        return tokens
