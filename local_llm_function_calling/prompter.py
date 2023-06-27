"""Prompter protocol for function calling with open source models"""

import json
from typing import Literal, NotRequired, Protocol, TypedDict

JsonType = str | int | float | bool | None | list["JsonType"] | dict[str, "JsonType"]


class FunctionParameters(TypedDict):
    """Function parameters"""

    type: Literal["object"]
    properties: dict[str, JsonType]


class FunctionType(TypedDict):
    """Function type"""

    name: str
    description: NotRequired[str]
    parameters: FunctionParameters


class FunctionCall(TypedDict):
    """Function call"""

    name: str
    parameters: JsonType


class TextPrompter(Protocol):
    """Prompter protocol for function calling with open source models"""

    def prompt(
        self,
        prompt: str,
        functions: list[FunctionType],
        function_to_call: str | None = None,
    ) -> str:
        """Prompt the user for input

        If function_to_call is None, then the prompt's aim should be to select
        the correct function to call. If function_to_call is not None, then the
        prompt's aim should be to generate the correct arguments for the
        function.

        Args:
            prompt (str): The prompt for the AI
            functions (list[FunctionType]): The functions to choose from
            function_to_call (str | None): The function to call.
                When None, the prompt should be to select the function to call.
        """
        ...  # pylint: disable=unnecessary-ellipsis


class CompletionModelPrompter:
    """Basic text prompter"""

    def prompt_for_function(self, function: FunctionType) -> str:
        """Generate the prompt section for a function"""
        header = (
            f"{function['name']} - {function['description']}"
            if "description" in function
            else function["name"]
        )
        schema = json.dumps(function["parameters"]["properties"], indent=4)
        packed_schema = f"```jsonschema\n{schema}\n```"
        return f"{header}\n{packed_schema}"

    def prompt_for_functions(self, functions: list[FunctionType]) -> str:
        return "\n\n".join(
            [self.prompt_for_function(function) for function in functions]
        )

    @property
    def head(self) -> str:
        """The head of the prompt"""
        return "\n\nAvailable functions:\n"

    @property
    def call_header(self) -> str:
        """The header for the function call"""
        return "\n\nFunction call: "

    def function_call(self, function_to_call: str | None = None) -> str:
        """Create a function call prompt"""
        return self.call_header + (
            f"{function_to_call}\n```json\n" if function_to_call else ""
        )

    def prompt(
        self,
        prompt: str,
        functions: list[FunctionType],
        function_to_call: str | None = None,
    ) -> str:
        """Create a function call prompt"""
        available_functions = self.prompt_for_functions(functions)
        return (
            prompt
            + self.head
            + available_functions
            + self.call_header
            + self.function_call(function_to_call)
        )


class InstructModelPrompter(CompletionModelPrompter):
    """Basic prompter for instruct models"""

    @property
    def head(self) -> str:
        """The head of the prompt"""
        return (
            "Your task is to call a function when needed. "
            "You will be provided with a list of functions. "
            "Available functions:\n"
        )

    def prompt(
        self,
        prompt: str,
        functions: list[FunctionType],
        function_to_call: str | None = None,
    ) -> str:
        """Create a function call prompt"""
        available_functions = self.prompt_for_functions(functions)
        return (
            self.head
            + available_functions
            + "\n\n"
            + prompt
            + self.function_call(function_to_call)
        )
