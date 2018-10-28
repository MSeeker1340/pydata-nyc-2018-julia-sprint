# PyDate NYC 2018 - Julia Sprint

## Better Julia lexer for Pygments

The improved lexer is contained in `julia.py`, which is updated with the latest reserved keywords and exported names from `Core` and `Base` for Julia v1.0. The lexer still could use some improvement in regards to context-aware identification of function names and types.

### How to use

Refer to the official [Pygments documentation](http://pygments.org/) for more information on how to use the package. To use the new lexer, just import `JuliaLexer` from `julia.py` instead of the pygments package:

```python
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.formatters import LatexFormatter
from julia import JuliaLexer

# Snippet from LinearAlgebra
code = '''
function triu!(M::AbstractMatrix, k::Integer)
    @assert !has_offset_axes(M)
    m, n = size(M)
    for j in 1:min(n, m + k)
        for i in max(1, j - k + 1):m
            M[i,j] = zero(M[i,j])
        end
    end
    M
end
'''

html = highlight(code, JuliaLexer(), HtmlFormatter(full=True))
latex = highlight(code, JuliaLexer(), LatexFormatter(full=True))
```

You can then save the outputs to a file. If in a notebook environment, you can `from IPython.core.display import HTML` then `HTML(html)` to view the html output.

### Julia script for printing out global names

`print_names.jl` is the script we used to print out all the names exported by `Core` and `Base`, which could be used after a major Julia update to modify the lexer with changes to the global namespace. The script is modified from the [Julia Sublime Text plugin](https://github.com/JuliaEditorSupport/Julia-sublime). It is not very robust though, e.g. names like `pi` and `e` get incorrectly classified, so care must be given when using the script.
