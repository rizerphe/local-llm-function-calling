# Constrained generation

You can also use the [Constrainer](local_llm_function_calling.Constrainer) class to just generate text based on constraints. You then have two options: either using a [builtin JSON schema constraint](local_llm_function_calling.JsonSchemaConstraint) or a custom one.

## JSON schema

You can generate based on a simple JSON schema. Note that this does not support the full jsonschema specification, but instead uses a simplified format similar to that of OpenAI. Here's a simple usage example:

```python
from local_llm_function_calling import Constrainer, JsonSchemaConstraint


schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "maxLength": 10},
        "age": {"type": "integer"}
    },
    "enforceOrder": ["name", "age"]
}

constraint = JsonSchemaConstraint(schema)
constrainer = Constrainer("gpt2")
raw_json = constrainer.generate("Prefix.\n", constraint, max_len=100)
truncated_json = raw_json[:constraint.validate(raw_json).end_index]
```

<details><summary>This is the generated JSON:</summary>

```json
{
  "name": "TheTheThe.",
  "age": -1
}
```

Note that gpt2 was used, so it's irrational to expect high quality output.

</details>

`raw_json` can containe extra characters at the end, that's why we then create `truncated_json`. The prefix will be prepended to the generated data and then used as the prompt for the model.

## Custom constraints

If you don't want the output to just adhere to a JSON schema, you can also define your own constraints. A constraint is just a callable that takes in the generated text and checks whether what's been generated is valid and whether it's the complete output. Here's a simple example that forces the output to be all-lowercase.

```python
def lowercase_sentence_constraint(text: str):
    # Has to return (is_valid, is_complete)
    return [text.islower(), text.endswith(".")]

constrainer = Constrainer("gpt2")

generated = constrainer.generate("Prefix.\n", lowercase_sentence_constraint, max_len=10)
```
