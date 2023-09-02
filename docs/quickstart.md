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
generator = Generator.hf(functions, "gpt2")
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

## Llama.cpp

The [meta's llama2 family of models](https://ai.meta.com/llama/) (especially codellama) are so much more suited for this task than most other open source models. Far from everyone has the resources required to run the models as is though. One of the solutions is quantization. Quantized models are smaller and require way fewer resources, but produce lower quality results. This tool supports [llama.cpp](https://github.com/ggerganov/llama.cpp), which allows you to run these quantized models.

To use llama.cpp, you have to install the project with:

```sh
pip install local-llm-function-calling[llama-cpp]
```

Then download one of the quantized models (e.g. one of [these](https://huggingface.co/TheBloke/CodeLlama-13B-Instruct-GGUF#provided-files)) and use [LlamaModel](local_llm_function_calling.model.llama.LlamaModel) to load it:

```python
from local_llm_function_calling.model.llama import LlamaModel

generator = Generator(
    functions,
    LlamaModel(
        "codellama-13b-instruct.Q6_K.gguf"
    ),
)
```
