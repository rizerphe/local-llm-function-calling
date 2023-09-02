# Generating a function call

The library provides a [Generator](local_llm_function_calling.Generator) class that's supposed to fully replace OpenAI's function calling. It combines the functionality of the different [prompters](local_llm_function_calling.TextPrompter) and a constrainer to generate a full function call, similar to what OpenAI does.

## Completion models

The most basic usage example is with a simple completion model. We'll use a huggingface model with a [CompletionModelPrompter](local_llm_function_calling.CompletionModelPrompter), which is its default, to construct the prompt, along with a `Generator` for it:

```python
from local_llm_function_calling import Generator
```

We need to specify the functions and the completion model to use - we'll use a simple get weather function and gpt2; if you need help generating schemas [another project of mine](https://github.com/rizerphe/openai-functions) can help. GPT2 is not a model I'd generally recommend because of it's tiny size - switch it out for a larger one if you want to do anything useful (codellama works really well).

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

generator = Generator.hf(functions, "gpt2")
```

You can pass in the huggingface model itself, or even both the model and the tokenizer, not just its name:

```python
Generator.hf(functions, model)
Generator.hf(functions, model, tokenizer)
```

These are shorthands for:

```python
from local_llm_function_calling.model.huggingface import HuggingfaceModel

Generator(functions, HuggingfaceModel(model))
Generator(functions, HuggingfaceModel(model, tokenizer))
```

When we have the generator ready, we can then pass in a prompt and have it construct a function call for us:

```python
function_call = generator.generate("What is the weather like today in Brooklyn?")
```

<details><summary>For the example above, GPT2 generates something like this:</summary>

```json
{
  "name": "get_current_weather",
  "parameters": "{\n    \"location\": \"{{{{{{{{{{{{{{{{{{{{\"\n}"
}
```

</details>

## Instruct models

You might want to use a different prompting scheme, however, for example when using an instruct model. For instruct models specifically, however, there's [InstructModelPrompter](local_llm_function_calling.InstructModelPrompter). Here's how to use it:

```python
from local_llm_function_calling import Generator, InstructModelPrompter


generator = Generator.hf(functions, "gpt2", prompter=InstructModelPrompter())
```

Then you use it the same way as you'd use the generator normally:

```python
function_call = generator.generate("What is the weather like today in Brooklyn?")
```

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

[LlamaModel](local_llm_function_calling.model.llama.LlamaModel) by default uses a prompt template compatible with the llama2 Instruct models.

## Your own prompters

I would suggest trying out different prompts; mine had very little thought put into them. You can do so by creating a class that adheres to the [TextPrompter](local_llm_function_calling.TextPrompter) protocol and passing them in as the prompter. A prompter has to define a `prompt(prompt: str, functions: list[local_llm_function_calling.prompter.FunctionType], function_to_call: str | None = None) → PromptType` method. If it has a `function_to_call` provided, it should return the full prefix for the call schema generation; otherwise it should return the prefix for the function name generation. `PromptType` depends on the model - for huggingface models it's a string, for llama.cpp models it's an array of integers and bytes objects; integers to allow for special tokens to be passed (a `bos` token is expected to go first).
