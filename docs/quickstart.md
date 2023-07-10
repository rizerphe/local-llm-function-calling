# Quickstart

This is a tool that allows you to replicate OpenAI's [function calling](https://openai.com/blog/function-calling-and-other-api-updates) feature with local models while enforcing the output schema.

## Installation

```shell
pip install local-llm-function-calling
```

## Usage

Import the generator:

```python
from local_llm_function_calling import Generator
```

Define your functions ([another project of mine](https://github.com/rizerphe/openai-functions) can help):

```python
functions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                    "maxLength": 20,
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    }
]
```

Initialize the generator with the Hugging Face model, tokenizer, and functions:

```python
generator = Generator(functions, "gpt2")
```

Generate text using a prompt:

```python
function_call = generator.generate("What is the weather like today in Brooklyn?")
print(function_call)
```

<details><summary>The output:</summary>

```json
{
  "name": "get_current_weather",
  "parameters": "{\n    \"location\": \"{{{{{{{{{{{{{{{{{{{{\"\n}"
}
```

</details>
