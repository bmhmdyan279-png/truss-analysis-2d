from abc import ABC, abstractmethod


class ITranslator(ABC):
    @abstractmethod
    def translate(self, key: str, **kwargs) -> str:
        pass
