# Prompts

This directory contains the ParEval prompts.
The prompts for the generation task are contained in `generation-prompts.json`
and the prompts for the translation task are in `translation-prompts.json`.


The format of the prompts dataset is as follows:

```json
[
    {
        "problem_type": "stencil",
        "language": "cpp",
        "name": "17_problem_name",
        "parallelism_model": "serial",
        "prompt": "/* prompt for the model here */\nvoid foo(int x) {"
    },
    ...
]
```

## Other Utilities

`create-serial-tests.py` -- this script will parse out the sequential baselines
for each problem from the drivers and create a "fake" output file with these
as the sequential solutions. This can be used to test the driver setup and make
sure it's working.

`count-tokens.py` -- this can be used to estimate the number of tokens passed
to the OpenAI API and, thus, estimate the cost of generated outputs.