from typing import Text


class NameMixIn:
    """
    Add class name as `name` attribute
    """
    @property
    def name(self) -> Text:
        return self.__class__.__name__
