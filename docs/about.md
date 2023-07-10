# How It Works

The tool leverages the power of large language models to generate text while enforcing a JSON schema. To understand how it works, let's first explore the basics of large language models and tokens.

Large language models, are trained on vast amounts of text data to learn the statistical patterns and structures of language. These models are capable of generating coherent and contextually relevant text given a prompt or a partial sentence. In this project, the focus is on using such models to generate text that adheres to a specified JSON schema, or to a different constraint provided by the developer.

In the context of language models, a token is the fundamental unit of text. It can represent a single character, a word, or even a subword, depending on the tokenization approach used. For example, in English, a token can correspond to a word like "cat" or a subword like "un" and "happy". Tokens are the building blocks that language models operate on, and they carry semantic and syntactic information.

When generating text, language models typically predict the next token based on the context provided by the preceding tokens. The probability distribution over the vocabulary of tokens is used to determine the likelihood of different tokens occurring next. Higher probabilities indicate more probable tokens based on the training data. When generating text with a prompt like "What is the weather like", the language model examines the preceding tokens (e.g., "What is the weather like") to predict the next token. In this case, the model might assign higher probabilities to tokens like "in" or "today" based on the patterns it has learned during training. It considers the likelihood of different tokens given the context to generate a coherent and contextually appropriate continuation.

In the "local-llm-function-calling" project, generating text goes beyond selecting tokens solely based on their likelihood. It incorporates the additional constraint of adhering to a given JSON schema, or another user-provided constraint. The schema defines the structure, properties, and constraints of the data that should be generated. This means that even a model that just generates completely random text normally will still generate valid JSON as the output.

During text generation, the `Constrainer` class constructs the text by iteratively adding tokens to a prefix. It generates tokens according to their likelihood, as suggested by the language model, but also checks whether each token, when appended to the generated text, adheres to the JSON schema. This is achieved by passing the generated text plus each candidate token to the constraint function.

For example, given the schema and output:

Schema:

```json
{
  "type": "object",
  "properties": {
    "location": {
      "type": "string",
      "maxLength": 20
    },
    "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] }
  },
  "required": ["location"]
}
```

Output:

```json
{
    "location": "San Francisco, CA",
    "
```

If the model suggests the next token being `unit`, it will be accepted, and if it suggests, for example, `date`, it won't be accepted as `date` is not a property defined in the schema. The tool selects a token with the highest likelihood that still adheres to the schema.
