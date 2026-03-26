from abc import ABC, abstractmethod

class ConfigureScheduler(ABC):
    """This is an example of what a scheduler constructor."""
    @abstractmethod
    def __init__(self, optimzer):
        pass