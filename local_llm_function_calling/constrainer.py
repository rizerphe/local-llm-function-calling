"""A generator for the responses to a function call"""
from __future__ import annotations
from itertools import count
from typing import Callable, Generic, TYPE_CHECKING, TypeVar

import json_schema_enforcer

from .exceptions import InvalidSchemaError, NoValidTokensError
from .exceptions import SequenceTooLongError

if TYPE_CHECKING:
    from .model import Model
    from local_llm_function_calling.model.common import Generation


class JsonSchemaConstraint:
    """A JSON schema constraint"""

    def __init__(
        self, schema: dict, style: json_schema_enforcer.StyleConfig | None = None
    ) -> None:
        """Create a JSON schema constraint

        Args:
            schema (dict): The schema to use
            style (json_schema_enforcer.StyleConfig | None): The style to use
                (specifies indentation, etc.)

        Raises:
            InvalidSchemaError: The schema is invalid
        """
        if style is None:
            self.style = json_schema_enforcer.StyleConfig(True, 4, True, 0, 0)
        else:
            self.style = style
        parser = json_schema_enforcer.parser_for_schema(schema)
        if parser is None:
            raise InvalidSchemaError()
        self.parser = parser

    def validate(self, text: str) -> json_schema_enforcer.schema.ValidationResult:
        """Validate the text against the schema

        Args:
            text (str): The text to validate

        Returns:
            json_schema_enforcer.schema.ValidationResult: The validation result
        """
        return self.parser.validate(text, style_config=self.style)

    def __call__(self, text: str) -> tuple[bool, bool]:
        """Validate the text against the schema

        Args:
            text (str): The text to validate

        Returns:
            tuple[bool, bool]: A tuple of (is_valid, is_complete)
        """
        result = self.validate(text)
        return result.valid, result.end_index is not None


class EnumConstraint:
    """An enum constraint, allowing only a set of values"""

    def __init__(self, values: list[str], full_generation: bool = True) -> None:
        """Create an enum constraint

        Args:
            values (list[str]): The values to allow
            full_generation (bool): Whether to require full generation,
                or just that the generated value is a prefix of one of the
                values
        """
        self.values = values
        self.full_generation = full_generation

        if any(
            (value.startswith(prefix) and value != prefix)
            for value in values
            for prefix in values
        ):
            raise ValueError("Values must not be prefixes of each other")

    def __call__(self, text: str) -> tuple[bool, bool]:
        """Validate the text against the schema

        Args:
            text (str): The text to validate

        Returns:
            tuple[bool, bool]: A tuple of (is_valid, is_complete)
        """
        fitting = self.fitting(text)
        is_valid = any(fitting)
        is_complete = (
            any(text.startswith(value) for value in self.values)
            if self.full_generation
            else (len(fitting) == 1)
        )
        return is_valid, is_complete

    def fitting(self, text: str) -> list[str]:
        """Get the fitting values for the text

        Args:
            text (str): The text to check

        Returns:
            list[str]: The fitting values
        """
        return [
            value
            for value in self.values
            if value.startswith(text) or text.startswith(value)
        ]


PrefixType = TypeVar("PrefixType")


class Constrainer(Generic[PrefixType]):
    """Generate text with an LLM in a constrained way"""

    def __init__(
        self,
        model: Model[PrefixType],
    ) -> None:
        """Create a constrainer for generating text with an LLM

        Args:
            model (Model): The model to use
        """
        self.model = model

    def gen_next_token(
        self,
        generation: Generation,
        constraint: Callable[[str], tuple[bool, bool]],
    ) -> tuple[bool, int]:
        """Generate the next token and register it

        Args:
            generation (Generation): The generation to use
            constraint (Callable[[str], tuple[bool, bool]]):
                A function that takes a string and returns a tuple of
                (is_valid, is_complete)

        Raises:
            NoValidTokensError: There are no valid tokens to generate

        Returns:
            tuple[bool, int]: A tuple, the first element is whether the
                generation is complete, the second is the number of characters
                generated so far (or 0 if the generation is complete)
        """
        try:
            sorted_tokens = generation.get_sorted_tokens()
        except SequenceTooLongError:
            return (True, 0)
        for token in sorted_tokens:
            generated = generation.get_generated(token)
            fit = constraint(generated)
            if fit[0]:
                generation.register_token(token)
                if fit[1]:
                    return (True, 0)
                return (False, len(generated))
        raise NoValidTokensError()

    def advance_generation(
        self,
        generation: Generation,
        constraint: Callable[[str], tuple[bool, bool]],
        max_len: int | None = None,
    ) -> bool:
        """Advance the generation by one token

        Args:
            generation (Generation): The generation to use
            constraint (Callable[[str], tuple[bool, bool]]):
                A function that takes a string and returns a tuple of
                (is_valid, is_complete)
            max_len (int | None): The maximum length of the generated string

        Returns:
            bool: Whether the generation is complete
        """
        done, length = self.gen_next_token(generation, constraint)
        if done:
            return True
        return max_len is not None and length >= max_len

    def generate(
        self,
        prefix: PrefixType,
        constraint: Callable[[str], tuple[bool, bool]],
        max_len: int | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        """Generate a string with the LLM

        Args:
            prefix: The prefix to use; the type depends on the model
            constraint (Callable[[str], tuple[bool, bool]]):
                A function that takes a string and returns a tuple of
                (is_valid, is_complete)
            max_len (int | None): The maximum length of the generated string
            max_new_tokens (int | None): The maximum number of tokens to generate

        Raises:
            NoValidTokensError: There are no valid tokens to generate

        Returns:
            str: The generated value
        """
        generation = self.model.start_generation(prefix)
        for _ in range(max_new_tokens) if max_new_tokens else count():
            if self.advance_generation(generation, constraint, max_len):
                break
        return generation.get_generated()
