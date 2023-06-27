"""A generator for the responses to a function call"""
from typing import Callable, Iterator

import json_schema_enforcer
from transformers import AutoModelForCausalLM, AutoTokenizer


class ConstrainerError(Exception):
    """An error in the constrainer"""


class SequenceTooLong(ConstrainerError):
    """The sequence is too long to generate"""


class NoValidTokens(ConstrainerError):
    """There are no valid tokens to generate"""


class InvalidSchema(ConstrainerError):
    """The schema is invalid"""


class JsonSchemaConstraint:
    """A JSON schema constraint"""

    def __init__(
        self, schema: dict, style: json_schema_enforcer.StyleConfig | None = None
    ):
        """Create a JSON schema constraint

        Args:
            schema (dict): The schema to use
            style (json_schema_enforcer.StyleConfig | None): The style to use
                (specifies indentation, etc.)
        """
        if style is None:
            self.style = json_schema_enforcer.StyleConfig(True, 4, True, 0, 0)
        else:
            self.style = style
        parser = json_schema_enforcer.parser_for_schema(schema)
        if parser is None:
            raise InvalidSchema()
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


class Constrainer:
    """Generate text with an LLM in a constrained way"""

    def __init__(
        self,
        model: AutoModelForCausalLM | str,
        tokenizer: AutoTokenizer | None = None,
    ):
        """Create a constrainer for generating text with an LLM

        Args:
            model (AutoModelForCausalLM | str): The model to use for generation
            tokenizer (AutoTokenizer | None): The tokenizer to use.
                Automatically loaded if not provided.
        """
        if isinstance(model, str):
            self.model = AutoModelForCausalLM.from_pretrained(model)
        else:
            self.model = model
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model.name_or_path)
        else:
            self.tokenizer = tokenizer

    def get_sorted_tokens(self, prompt: str) -> Iterator[str]:
        """Get the tokens sorted by their likelihood

        Args:
            prompt (str): The prompt to use

        Raises:
            SequenceTooLong: The input sequence is too long

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
            if token == self.tokenizer.eos_token_id:
                # Don't yield the EOS token
                continue
            yield self.tokenizer.decode(token)

    def generate(
        self,
        prefix: str,
        constraint: Callable[[str], tuple[bool, bool]],
        max_len: int | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        """Generate one of the values in an enum, for choosing the function

        Args:
            prefix (str): The prefix to use
            constraint (Callable[[str], tuple[bool, bool]]):
                A function that takes a string and returns a tuple of
                (is_valid, is_complete)
            max_len (int | None): The maximum length of the generated string
            max_new_tokens (int | None): The maximum number of tokens to generate

        Raises:
            NoValidTokens: There are no valid tokens to generate

        Returns:
            str: The generated value
        """
        generated = ""
        tokens = 0
        while True:
            try:
                sorted_tokens = self.get_sorted_tokens(prefix)
            except SequenceTooLong:
                return generated
            for token in sorted_tokens:
                fit = constraint(generated + token)
                if fit[0]:
                    generated += token
                    tokens += 1
                    if fit[1]:
                        return generated
                    break
            else:  # For loop did not break
                raise NoValidTokens()
            if max_len is not None and len(generated) >= max_len:
                return generated
            if max_new_tokens is not None and tokens >= max_new_tokens:
                return generated
