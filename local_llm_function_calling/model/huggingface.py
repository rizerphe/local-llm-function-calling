"""A container for huggingface models specifically"""
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..exceptions import SequenceTooLongError
from ..prompter import CompletionModelPrompter, TextPrompter


class HuggingfaceGeneration:
    """A single generation sequence with a huggingface model"""

    def __init__(
        self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prefix: str
    ) -> None:
        """Create a generation sequence

        Args:
            model (AutoModelForCausalLM): The model to use for generation
            tokenizer (AutoTokenizer): The tokenizer to use
            prefix (str): The generation prefix
        """
        self.model = model
        self.tokenizer = tokenizer
        self.inputs = self.tokenizer(prefix, return_tensors="pt")["input_ids"]
        self.generated: list[int] = []
        self.candidates: torch.Tensor | None = None

    def get_sorted_tokens(self) -> Iterator[int]:
        """Get the tokens sorted by probability

        Raises:
            SequenceTooLongError: If the sequence is too long to generate

        Yields:
            The next of the most likely tokens
        """
        if self.inputs.shape[1] >= self.model.config.n_positions:
            raise SequenceTooLongError()
        gen_tokens = self.model.generate(
            input_ids=self.inputs,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        tokens_rated = gen_tokens.scores[0].argsort(descending=True)[0]
        for token in tokens_rated:
            if token == self.tokenizer.eos_token_id:
                # Don't yield the EOS token
                continue
            yield token

    def register_token(self, token: int) -> None:
        """Select the token for this generation step

        Args:
            token (int): The token to select
        """
        self.generated.append(token)
        self.inputs = torch.cat([self.inputs, torch.tensor([[token]])], dim=1)

    def get_generated(self, candidate: int | None = None) -> str:
        """Get the generated sequence

        Args:
            candidate (int | None): The token to add to the sequence

        Returns:
            str: The generated sequence
        """
        return self.tokenizer.decode(
            self.generated + ([candidate] if candidate else [])
        )


class HuggingfaceModel:
    """A container for a huggingface model"""

    def __init__(
        self,
        model: AutoModelForCausalLM | str,
        tokenizer: AutoTokenizer | str | None = None,
    ) -> None:
        """Create a huggingface model

        Args:
            model (AutoModelForCausalLM | str): The model to use for generation
            tokenizer (AutoTokenizer | str | None): The tokenizer to use.
                Automatically loaded if not provided.
        """
        if isinstance(model, str):
            self.model = AutoModelForCausalLM.from_pretrained(model)
        else:
            self.model = model
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        elif tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model.name_or_path)
        else:
            self.tokenizer = tokenizer

    def start_generation(self, prefix: str) -> HuggingfaceGeneration:
        """Start a new generation sequence

        Args:
            prefix (str): The generation prefix

        Returns:
            HuggingfaceGeneration: The generation sequence initialized with the
                prefix
        """
        return HuggingfaceGeneration(self.model, self.tokenizer, prefix)

    def default_prompter(self) -> TextPrompter[str, str]:
        """Get the default prompter for this model

        Returns:
            A generic CompletionModelPrompter
        """

        return CompletionModelPrompter()
