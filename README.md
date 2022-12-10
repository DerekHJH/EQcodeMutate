# Equivalent Code Mutation Engine

## Build `tree-sitter`

We use `tree-sitter` to parse code snippets and extract variable names. You need to go to `./python_parser/parser_folder` folder and build tree-sitter using the following commands:

```
bash build.sh
```

## Usage

The main driver code is in "EQCodeMutate.py". Try "python EQCodeMutate.py" in your command line, and you will get a json file that contains the mutated code. Refer to "EQCodeMutate.py" for more details about usage.

The preferable usage is to import the get_EQCodeMutate function and use this function in your own code.