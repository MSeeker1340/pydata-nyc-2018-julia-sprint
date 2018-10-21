open("names.txt", "w") do f

    # Script modified from
    # https://github.com/JuliaEditorSupport/Julia-sublime/blob/master/Julia.sublime-syntax
    base_types = join(map(x -> "'$x'", sort(unique((
        filter(x -> isletter(x[1]), 
            string.(filter!(x -> isa(eval(x), DataType) || isa(eval(x), UnionAll),
                filter!(x -> !Base.isdeprecated(Base, x),
                    [names(Base); names(Core)])))))))), ',')
    base_funcs = join(map(x -> "'$x'",
        filter!(x -> isascii(x[1]) && isletter(x[1]) && islowercase(x[1]),
            map(string, filter!(x -> !Base.isdeprecated(Base, x),
                [names(Base); names(Core); :include])))), ',')
    base_macros = join(map(x -> "'$x'",
    filter!(x -> x[1] == '@',
        map(string, filter!(x -> !Base.isdeprecated(Base, x),
            [names(Base); names(Core)])))), ',')
    base_modules = join(map(x -> "'$x'",
        filter!(x -> isa(eval(x), Module) && !Base.isdeprecated(Base, x),
            [names(Base); names(Core)])), ",")
    
    println(f, "Core & Base types:")
    println(f, base_types)
    println(f, "\nCore & Base functions:")
    println(f, base_funcs)
    println(f, "\nCore & Base macros")
    println(f, base_macros)
    println(f, "\nCore & Base modules:")
    println(f, base_modules)
end
