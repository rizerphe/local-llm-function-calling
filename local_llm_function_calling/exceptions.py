"""Exceptions for the function calling library"""


class FunctionCallingError(Exception):
    """An error in the function calling library"""


class ConstrainerError(FunctionCallingError):
    """An error in the constrainer"""


class NoValidTokensError(ConstrainerError):
    """There are no valid tokens to generate"""


class InvalidSchemaError(ConstrainerError):
    """The schema is invalid"""


class GenerationError(FunctionCallingError):
    """An error in the generation"""


class SequenceTooLongError(GenerationError):
    """The sequence is too long to generate"""
