To re-create API docs:

1. Install `pdoc` package (do not confuse with `pdoc3`) if not yet installed
    ```commandline
    conda install pdoc
    ```
2. Run `pdoc` taking into account reStructured docstring syntax
    ```commandline
    cd [path_to_fermatrica_rep_directory]
    pip install . 
    pdoc -d restructuredtext -o docs/api fermatrica_rep
    ```
