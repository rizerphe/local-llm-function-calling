"""An abstract definition of a model."""
from __future__ import annotations
from typing import Any, Iterator, Protocol, TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from ..prompter import TextPrompter


TokenType = TypeVar("TokenType")


class Generation(Protocol[TokenType]):
    """One single generation"""

    def get_sorted_tokens(self) -> Iterator[TokenType]:
        """Get the tokens sorted by probability"""
        ...

    def register_token(self, token: TokenType) -> None:
        """Select the token for this generation step

        Args:
            token (TokenType): The token to select
        """

    def get_generated(self, candidate: TokenType | None = None) -> str:
        """Get the generated sequence so far

        Args:
            candidate (int | None): The token to add to the sequence
                (should be one of the tokens returned by
                get_sorted_tokens in this generation step)
        """
        ...


PrefixType = TypeVar("PrefixType")


class Model(Protocol[PrefixType]):
    """A container for a generic language model"""

    def start_generation(self, prefix: PrefixType) -> Generation:
        """Start a new generation sequence

        Args:
            prefix (PrefixType): The generation prefix
        """
        ...

    def default_prompter(self) -> TextPrompter[PrefixType, Any]:
        """Get the default prompter for this model"""
        ...
