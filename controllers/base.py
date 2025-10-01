"""
Base controller class
"""


class BaseController:
    """Base class for all controllers"""

    def __init__(self, name="Controller"):
        self.name = name

    def control(self, *args, **kwargs):
        """
        Compute control action

        Args:
            *args: Controller-specific arguments
            **kwargs: Controller-specific keyword arguments

        Returns:
            Control action (format depends on controller type)
        """
        raise NotImplementedError("Subclasses must implement control()")

    def reset(self):
        """Reset controller state"""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
