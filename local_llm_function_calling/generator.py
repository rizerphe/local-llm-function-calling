"""A generator for the responses to a function call"""
from typing import Iterator

import json_schema_enforcer
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompter import CompletionModelPrompter, FunctionCall, FunctionType, TextPrompter


class SequenceTooLong(Exception):
    """The sequence is too long to generate"""


class Generator:
    """Generate the function call based on the schema"""

    def __init__(
        self,
        functions: list[FunctionType],
        model: AutoModelForCausalLM | str,
        tokenizer: AutoTokenizer | None = None,
        prompter: TextPrompter | None = None,
    ):
        """Create a generator for the responses to a function call

        Args:
            functions (list[FunctionType]): The functions to use.
            model (AutoModelForCausalLM | str): The model to use for generation
            tokenizer (AutoTokenizer | None): The tokenizer to use.
                Automatically loaded if not provided.
            prompter (TextPrompter, optional): The prompter to use.
                Will use CompletionModelPrompter if not provided.
        """
        if isinstance(model, str):
            self.model = AutoModelForCausalLM.from_pretrained(model)
        else:
            self.model = model
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model.name_or_path)
        else:
            self.tokenizer = tokenizer
        self.prompter: TextPrompter = prompter or CompletionModelPrompter()
        self.functions = functions or []

    def get_sorted_tokens(self, prompt: str) -> Iterator[str]:
        """Get the tokens sorted by their likelihood

        Args:
            prompt (str): The prompt to use

        Yields:
            str: The tokens sorted by their likelihood
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if inputs["input_ids"].shape[1] >= self.model.config.n_positions:
            raise SequenceTooLong()
        gen_tokens = self.model.generate(
            **inputs,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=1,
            pad_token_id=self.tokenizer.eos_token_id
        )
        tokens_rated = gen_tokens.scores[0].argsort(descending=True)[0]
        for token in tokens_rated:
            yield self.tokenizer.decode(token)

    def _begins_enum(self, prefix: str, allowed: list[str]) -> bool:
        """Check if the prefix begins one of the allowed values,
        for choosing the function to call

        Args:
            prefix (str): The prefix to check
            allowed (list[str]): The allowed values

        Returns:
            bool: Whether the prefix begins one of the allowed values
        """
        for item in allowed:
            if item.startswith(prefix) or prefix.startswith(item):
                return True
        return False

    def _generate_allowed_in_enum(self, prefix: str, allowed: list[str]) -> str:
        """Generate one of the values in an enum, for choosing the function

        Args:
            prefix (str): The prefix to use
            allowed (list[str]): The allowed values

        Returns:
            str: The generated value
        """
        generated = ""
        while True:
            for item in allowed:
                if item.startswith(generated):
                    return item
            for token in self.get_sorted_tokens(prefix):
                if self._begins_enum(generated + token, allowed):
                    generated += token
                    break

    def _choose_function(self, prompt: str) -> str:
        """Choose a function to call

        Args:
            prompt (str): The prompt to use
        """
        prefix = self.prompter.prompt(prompt, self.functions)
        return self._generate_allowed_in_enum(
            prefix, [function["name"] for function in self.functions]
        )

    def choose_function(self, prompt: str, function_call: str | None = None) -> str:
        """Choose a function to call

        Args:
            prompt (str): The prompt to use
            function_call (str | None): The function to call
                Will be generated if not provided.

        Returns:
            str: The function to call
        """
        if function_call is None:
            return self._choose_function(prompt)
        return function_call

    def generate_arguments(
        self,
        prompt: str,
        function_call: str,
        max_new_tokens: int | None = None,
        max_length: int | None = None,
    ) -> str:
        """Generate the arguments for the function

        Args:
            prompt (str): The prompt to use
            function_call (str): The function to call
            max_new_tokens (int | None): The maximum number of tokens to generate
            max_length (int | None): The maximum length of the generated sequence

        Returns:
            str: The arguments for the function, as a JSON string
                (may not be complete)
        """
        config = json_schema_enforcer.StyleConfig(True, 4, True, 0, 0)
        parser = json_schema_enforcer.parser_for_schema(
            [
                function
                for function in self.functions
                if function["name"] == function_call
            ][0][
                "parameters"
            ]  # type: ignore
        )
        if parser is None:
            raise ValueError("No parser found for arguments")
        prefix = self.prompter.prompt(prompt, self.functions, function_call)
        generated = ""
        generated_tokens = 0
        while True:
            try:
                for token in self.get_sorted_tokens(prefix + generated):
                    validated = parser.validate(generated + token, style_config=config)
                    if validated.valid:
                        if validated.end_index is not None:
                            return (generated + token)[: validated.end_index]
                        generated += token
                        break
            except SequenceTooLong:
                return generated
            generated_tokens += 1
            if max_new_tokens is not None and generated_tokens >= max_new_tokens:
                return generated
            if max_length is not None and len(generated) >= max_length:
                return generated

    def generate(
        self,
        prompt: str,
        function_call: str | None = None,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
    ) -> FunctionCall:
        """Generate the function call

        Args:
            prompt (str): The prompt to use
            function_call (str | None): The function call to use.
                Will be generated if not provided.
        """
        function_name = self.choose_function(prompt, function_call)
        arguments = self.generate_arguments(
            prompt, function_name, max_new_tokens, max_length
        )
        return {"name": function_name, "parameters": arguments}
