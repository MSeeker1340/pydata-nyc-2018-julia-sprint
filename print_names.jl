# Script modified from
# https://github.com/JuliaEditorSupport/Julia-sublime/blob/master/Julia.sublime-syntax
symbols = Set(filter(x -> !Base.isdeprecated(Base, x), [names(Base); names(Core)]))
types = filter(x -> isa(eval(x), DataType) || isa(eval(x), UnionAll), symbols)
funcs = filter(x -> (c = string(x)[1];
                     isascii(c) && isletter(c) && islowercase(c)), symbols)
macros = filter(x -> string(x)[1] == '@', symbols)
modules = filter(x -> isa(eval(x), Module), symbols)
others = setdiff(symbols, types, funcs, macros, modules)

function print_category(f::IO, cat)
    out = join(["'$(string(symbol))'" for symbol in cat], ',')
    println(f, out)
end

open("names.txt", "w") do f
    println(f, "Core & Base types:")
    print_category(f, types)
    println(f, "\nCore & Base functions:")
    print_category(f, funcs)
    println(f, "\nCore & Base macros")
    print_category(f, macros)
    println(f, "\nCore & Base modules:")
    print_category(f, modules)
    println(f, "\nOther names exported by Core & Base (handle these manually):")
    print_category(f, others)
end
