Welcome to local-llm-function-calling's documentation!
======================================================

The ``local-llm-function-calling`` project is designed to constrain the generation of Hugging Face text generation models by enforcing a JSON schema and facilitating the formulation of prompts for function calls, similar to OpenAI's `function calling <https://openai.com/blog/function-calling-and-other-api-updates>`_ feature, but actually enforcing the schema unlike OpenAI.

The project provides a ``Generator`` class that allows users to easily generate text while ensuring compliance with the provided prompt and JSON schema. By utilizing the ``local-llm-function-calling`` library, users can conveniently control the output of text generation models. It uses my own quickly sketched ``json-schema-enforcer`` project as the enforcer.

Features
--------

* Constrains the generation of Hugging Face text generation models to follow a JSON schema.
* Provides a mechanism for formulating prompts for function calls, enabling precise data extraction and formatting.
* Simplifies the text generation process through a user-friendly ``Generator`` class.

.. toctree::
   :maxdepth: 2
   :caption: Table of Contents:

   quickstart
   about
   generation
   constraining
   api

* :ref:`genindex`
* :ref:`search`

