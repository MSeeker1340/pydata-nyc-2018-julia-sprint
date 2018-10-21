# -*- coding: utf-8 -*-
"""
    pygments.lexers.julia
    ~~~~~~~~~~~~~~~~~~~~~

    Lexers for the Julia language.

    :copyright: Copyright 2006-2017 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

import re

from pygments.lexer import Lexer, RegexLexer, bygroups, do_insertions, \
    words, include
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
    Number, Punctuation, Generic
from pygments.util import shebang_matches, unirange

__all__ = ['JuliaLexer', 'JuliaConsoleLexer']

allowed_variable = (
    u'(?:[a-zA-Z_\u00A1-\uffff]|%s)(?:[a-zA-Z_0-9\u00A1-\uffff]|%s)*!*' %
    ((unirange(0x10000, 0x10ffff),) * 2))


class JuliaLexer(RegexLexer):
    """
    For `Julia <http://julialang.org/>`_ source code.

    .. versionadded:: 1.6
    """

    name = 'Julia'
    aliases = ['julia', 'jl']
    filenames = ['*.jl']
    mimetypes = ['text/x-julia', 'application/x-julia']

    flags = re.MULTILINE | re.UNICODE

    tokens = {
        'root': [
            (r'\n', Text),
            (r'[^\S\n]+', Text),
            (r'#=', Comment.Multiline, "blockcomment"),
            (r'#.*$', Comment),
            (r'[\[\]{}(),;]', Punctuation),

            # keywords
            (r'in\b', Keyword.Pseudo),
            (r'(true|false)\b', Keyword.Constant),
            (r'(local|global|const)\b', Keyword.Declaration),
            (r'(using|import|export)\b', Keyword.Namespace),
            (words([
                'begin', 'while', 'if', 'for', 'try', 'return', 'break', 'continue',
                'function', 'macro', 'quote', 'let', 'do', 'type', 'primitive',
                'struct', 'module', 'baremodule', 'using', 'import', 'export',
                'end', 'else', 'elseif', 'catch', 'finally'],
                suffix=r'\b'), Keyword.Reserved),

            # NOTE
            # Patterns below work only for definition sites and thus hardly reliable.
            #
            # functions
            # (r'(function)(\s+)(' + allowed_variable + ')',
            #  bygroups(Keyword, Text, Name.Function)),
            #
            # types
            # (r'(type|typealias|abstract|immutable)(\s+)(' + allowed_variable + ')',
            #  bygroups(Keyword, Text, Name.Class)),

            # types exported by Core and Base
            (words([
                'AbstractArray','AbstractChannel','AbstractChar','AbstractDict',
                'AbstractDisplay','AbstractFloat','AbstractIrrational','AbstractMatrix',
                'AbstractRange','AbstractSet','AbstractString','AbstractUnitRange',
                'AbstractVecOrMat','AbstractVector','Any','ArgumentError','Array',
                'AssertionError','BigFloat','BigInt','BitArray','BitMatrix','BitSet',
                'BitVector','Bool','BoundsError','CapturedException','CartesianIndex',
                'CartesianIndices','Cchar','Cdouble','Cfloat','Channel','Char','Cint',
                'Cintmax_t','Clong','Clonglong','Cmd','Colon','Complex','ComplexF16',
                'ComplexF32','ComplexF64','CompositeException','Condition','Cptrdiff_t',
                'Cshort','Csize_t','Cssize_t','Cstring','Cuchar','Cuint','Cuintmax_t',
                'Culong','Culonglong','Cushort','Cvoid','Cwchar_t','Cwstring','DataType',
                'DenseArray','DenseMatrix','DenseVecOrMat','DenseVector','Dict',
                'DimensionMismatch','Dims','DivideError','DomainError','EOFError','Enum',
                'ErrorException','Exception','ExponentialBackOff','Expr','Float16','Float32',
                'Float64','Function','GlobalRef','HTML','IO','IOBuffer','IOContext','IOStream',
                'IdDict','IndexCartesian','IndexLinear','IndexStyle','InexactError','InitError',
                'Int','Int128','Int16','Int32','Int64','Int8','Integer','InterruptException',
                'InvalidStateException','Irrational','KeyError','LinRange','LineNumberNode',
                'LinearIndices','LoadError','MIME','Matrix','Method','MethodError','Missing',
                'MissingException','Module','NTuple','NamedTuple','Nothing','Number',
                'OrdinalRange','OutOfMemoryError','OverflowError','Pair','PartialQuickSort',
                'PermutedDimsArray','Pipe','Ptr','QuoteNode','Rational','RawFD',
                'ReadOnlyMemoryError','Real','ReentrantLock','Ref','Regex','RegexMatch',
                'RoundingMode','SegmentationFault','Set','Signed','Some','StackOverflowError',
                'StepRange','StepRangeLen','StridedArray','StridedMatrix','StridedVecOrMat',
                'StridedVector','String','StringIndexError','SubArray','SubString',
                'SubstitutionString','Symbol','SystemError','Task','Text','TextDisplay',
                'Timer','Tuple','Type','TypeError','TypeVar','UInt','UInt128','UInt16',
                'UInt32','UInt64','UInt8','UndefInitializer','UndefKeywordError','UndefRefError',
                'UndefVarError','Union','UnionAll','UnitRange','Unsigned','Val','Vararg',
                'VecElement','VecOrMat','Vector','VersionNumber','WeakKeyDict','WeakRef'], suffix=r'\b'),
                Name.Variable.Class),

            # functions exported by Core and Base
            (words([
                'abs','abs2','abspath','accumulate','accumulate!','acos','acosd','acosh',
                'acot','acotd','acoth','acsc','acscd','acsch','adjoint','all','all!',
                'allunique','angle','any','any!','append!','argmax','argmin','ascii','asec',
                'asecd','asech','asin','asind','asinh','asyncmap','asyncmap!','atan','atand',
                'atanh','atexit','atreplinit','axes','backtrace','basename','big','bind',
                'binomial','bitstring','broadcast','broadcast!','bswap','bytes2hex',
                'bytesavailable','cat','catch_backtrace','cbrt','cd','ceil','cglobal',
                'checkbounds','checkindex','chmod','chomp','chop','chown','circcopy!',
                'circshift','circshift!','cis','clamp','clamp!','cld','close','cmp','coalesce',
                'code_lowered','code_typed','codepoint','codeunit','codeunits','collect',
                'complex','conj','conj!','convert','copy','copy!','copysign','copyto!','cos',
                'cosc','cosd','cosh','cospi','cot','cotd','coth','count','count_ones',
                'count_zeros','countlines','cp','csc','cscd','csch','ctime','cumprod','cumprod!',
                'cumsum','cumsum!','current_task','deepcopy','deg2rad','delete!','deleteat!',
                'denominator','detach','devnull','diff','digits','digits!','dirname',
                'disable_sigint','display','displayable','displaysize','div','divrem','download',
                'dropdims','dump','eachindex','eachline','eachmatch','eltype','empty','empty!',
                'endswith','enumerate','eof','eps','error','esc','escape_string','evalfile','exit',
                'exp','exp10','exp2','expanduser','expm1','exponent','extrema','factorial','falses',
                'fd','fdio','fetch','fieldcount','fieldname','fieldnames','fieldoffset','filemode',
                'filesize','fill','fill!','filter','filter!','finalize','finalizer','findall',
                'findfirst','findlast','findmax','findmax!','findmin','findmin!','findnext','findprev',
                'first','firstindex','fld','fld1','fldmod','fldmod1','flipsign','float','floatmax',
                'floatmin','floor','flush','fma','foldl','foldr','foreach','frexp','fullname',
                'functionloc','gcd','gcdx','gensym','get','get!','get_zero_subnormals','gethostname',
                'getindex','getkey','getpid','getproperty','gperm','hash','haskey','hasmethod',
                'hcat','hex2bytes','hex2bytes!','homedir','htol','hton','hvcat','hypot','identity',
                'ifelse','ignorestatus','im','imag','in','include_dependency','include_string',
                'indexin','insert!','instances','intersect','intersect!','inv','invmod','invperm',
                'invpermute!','isabspath','isabstracttype','isapprox','isascii','isassigned','isbits',
                'isbitstype','isblockdev','ischardev','iscntrl','isconcretetype','isconst','isdigit',
                'isdir','isdirpath','isdispatchtuple','isempty','isequal','iseven','isfifo','isfile',
                'isfinite','isimmutable','isinf','isinteger','isinteractive','isless','isletter',
                'islink','islocked','islowercase','ismarked','ismissing','ismount','isnan','isnumeric',
                'isodd','isone','isopen','ispath','isperm','ispow2','isprimitivetype','isprint',
                'ispunct','isqrt','isreadable','isreadonly','isready','isreal','issetequal','issetgid',
                'issetuid','issocket','issorted','isspace','issticky','isstructtype','issubnormal',
                'issubset','istaskdone','istaskstarted','istextmime','isuppercase','isvalid',
                'iswritable','isxdigit','iszero','iterate','join','joinpath','keys','keytype','kill',
                'kron','last','lastindex','lcm','ldexp','leading_ones','leading_zeros','length','lock',
                'log','log10','log1p','log2','lowercase','lowercasefirst','lpad','lstat','lstrip',
                'ltoh','macroexpand','map','map!','mapfoldl','mapfoldr','mapreduce','mapslices','mark',
                'match','max','maximum','maximum!','maxintfloat','merge','merge!','methods','min',
                'minimum','minimum!','minmax','missing','mkdir','mkpath','mktemp','mktempdir','mod',
                'mod1','mod2pi','modf','mtime','muladd','mv','nameof','names','ncodeunits','ndigits',
                'ndims','nextfloat','nextind','nextpow','nextprod','normpath','notify','ntoh','ntuple',
                'numerator','objectid','occursin','oftype','one','ones','oneunit','open','operm',
                'pairs','parent','parentindices','parentmodule','parse','partialsort','partialsort!',
                'partialsortperm','partialsortperm!','pathof','permute!','permutedims','permutedims!',
                'pi','pipeline','pointer','pointer_from_objref','pop!','popdisplay','popfirst!',
                'position','powermod','precision','precompile','prepend!','prevfloat','prevind',
                'prevpow','print','println','printstyled','process_exited','process_running','prod',
                'prod!','promote','promote_rule','promote_shape','promote_type','propertynames','push!',
                'pushdisplay','pushfirst!','put!','pwd','rad2deg','rand','randn','range','rationalize',
                'read','read!','readavailable','readbytes!','readchomp','readdir','readline',
                'readlines','readlink','readuntil','real','realpath','redirect_stderr','redirect_stdin',
                'redirect_stdout','redisplay','reduce','reenable_sigint','reim','reinterpret','relpath',
                'rem','rem2pi','repeat','replace','replace!','repr','reset','reshape','resize!',
                'rethrow','retry','reverse','reverse!','reverseind','rm','rot180','rotl90','rotr90',
                'round','rounding','rpad','rsplit','rstrip','run','schedule','searchsorted',
                'searchsortedfirst','searchsortedlast','sec','secd','sech','seek','seekend','seekstart',
                'selectdim','set_zero_subnormals','setdiff','setdiff!','setenv','setindex!','setprecision',
                'setproperty!','setrounding','show','showable','showerror','sign','signbit','signed',
                'significand','similar','sin','sinc','sincos','sind','sinh','sinpi','size','sizehint!',
                'sizeof','skip','skipchars','skipmissing','sleep','something','sort','sort!','sortperm',
                'sortperm!','sortslices','splice!','split','splitdir','splitdrive','splitext','sprint',
                'sqrt','stacktrace','startswith','stat','stderr','stdin','stdout','step','stride',
                'strides','string','strip','success','sum','sum!','summary','supertype','symdiff',
                'symdiff!','symlink','systemerror','take!','tan','tand','tanh','task_local_storage',
                'tempdir','tempname','textwidth','thisind','time','time_ns','timedwait','titlecase',
                'to_indices','touch','trailing_ones','trailing_zeros','transcode','transpose','trues',
                'trunc','truncate','trylock','tryparse','typeintersect','typejoin','typemax','typemin',
                'unescape_string','union','union!','unique','unique!','unlock','unmark','unsafe_copyto!',
                'unsafe_load','unsafe_pointer_to_objref','unsafe_read','unsafe_store!','unsafe_string',
                'unsafe_trunc','unsafe_wrap','unsafe_write','unsigned','uperm','uppercase',
                'uppercasefirst','valtype','values','vcat','vec','view','wait','walkdir','which','widemul',
                'widen','withenv','write','xor','yield','yieldto','zero','zeros','zip','applicable',
                'eval','fieldtype','getfield','ifelse','invoke','isa','isdefined','nfields','nothing',
                'setfield!','throw','tuple','typeassert','typeof','undef','include'
            ], suffix=r'\b'), Name.Function),

            # modules exported by Core and Base
            (words([
                'Base','Broadcast','Docs','GC','Iterators','Libc','MathConstants','Meta',
                'StackTraces','Sys','Threads','Core','Main'
            ], suffix=r'\b'), Name.Namespace),

            # builtins
            (words([
                u'ARGS', u'CPU_CORES', u'C_NULL', u'DevNull', u'ENDIAN_BOM',
                u'ENV', u'I', u'Inf', u'Inf16', u'Inf32', u'Inf64',
                u'InsertionSort', u'JULIA_HOME', u'LOAD_PATH', u'MergeSort',
                u'NaN', u'NaN16', u'NaN32', u'NaN64', u'OS_NAME',
                u'QuickSort', u'RoundDown', u'RoundFromZero', u'RoundNearest',
                u'RoundNearestTiesAway', u'RoundNearestTiesUp',
                u'RoundToZero', u'RoundUp', u'STDERR', u'STDIN', u'STDOUT',
                u'VERSION', u'WORD_SIZE', u'catalan', u'e', u'eu',
                u'eulergamma', u'golden', u'im', u'nothing', u'pi', u'γ',
                u'π', u'φ'],
                suffix=r'\b'), Name.Builtin),

            # operators
            # see: https://github.com/JuliaLang/julia/blob/master/src/julia-parser.scm
            (words([
                # prec-assignment
                u'=', u':=', u'+=', u'-=', u'*=', u'/=', u'//=', u'.//=', u'.*=', u'./=',
                u'\=', u'.\=', u'^=', u'.^=', u'÷=', u'.÷=', u'%=', u'.%=', u'|=', u'&=',
                u'$=', u'=>', u'<<=', u'>>=', u'>>>=', u'~', u'.+=', u'.-=',
                # prec-conditional
                u'?',
                # prec-arrow
                u'--', u'-->',
                # prec-lazy-or
                u'||',
                # prec-lazy-and
                u'&&',
                # prec-comparison
                u'>', u'<', u'>=', u'≥', u'<=', u'≤', u'==', u'===', u'≡', u'!=', u'≠',
                u'!==', u'≢', u'.>', u'.<', u'.>=', u'.≥', u'.<=', u'.≤', u'.==', u'.!=',
                u'.≠', u'.=', u'.!', u'<:', u'>:', u'∈', u'∉', u'∋', u'∌', u'⊆',
                u'⊈', u'⊂',
                u'⊄', u'⊊',
                # prec-pipe
                u'|>', u'<|',
                # prec-colon
                u':',
                # prec-plus
                u'+', u'-', u'.+', u'.-', u'|', u'∪', u'$',
                # prec-bitshift
                u'<<', u'>>', u'>>>', u'.<<', u'.>>', u'.>>>',
                # prec-times
                u'*', u'/', u'./', u'÷', u'.÷', u'%', u'⋅', u'.%', u'.*', u'\\', u'.\\', u'&', u'∩',
                # prec-rational
                u'//', u'.//',
                # prec-power
                u'^', u'.^',
                # prec-decl
                u'::',
                # prec-dot
                u'.',
                # unary op
                u'+', u'-', u'!', u'~', u'√', u'∛', u'∜'
            ]), Operator),

            # chars
            (r"'(\\.|\\[0-7]{1,3}|\\x[a-fA-F0-9]{1,3}|\\u[a-fA-F0-9]{1,4}|"
             r"\\U[a-fA-F0-9]{1,6}|[^\\\'\n])'", String.Char),

            # try to match trailing transpose
            (r'(?<=[.\w)\]])\'+', Operator),

            # strings
            (r'"""', String, 'tqstring'),
            (r'"', String, 'string'),

            # regular expressions
            (r'r"""', String.Regex, 'tqregex'),
            (r'r"', String.Regex, 'regex'),

            # backticks
            (r'`', String.Backtick, 'command'),

            # names
            (allowed_variable, Name),
            (r'@' + allowed_variable, Name.Decorator),

            # numbers
            (r'(\d+(_\d+)+\.\d*|\d*\.\d+(_\d+)+)([eEf][+-]?[0-9]+)?', Number.Float),
            (r'(\d+\.\d*|\d*\.\d+)([eEf][+-]?[0-9]+)?', Number.Float),
            (r'\d+(_\d+)+[eEf][+-]?[0-9]+', Number.Float),
            (r'\d+[eEf][+-]?[0-9]+', Number.Float),
            (r'0b[01]+(_[01]+)+', Number.Bin),
            (r'0b[01]+', Number.Bin),
            (r'0o[0-7]+(_[0-7]+)+', Number.Oct),
            (r'0o[0-7]+', Number.Oct),
            (r'0x[a-fA-F0-9]+(_[a-fA-F0-9]+)+', Number.Hex),
            (r'0x[a-fA-F0-9]+', Number.Hex),
            (r'\d+(_\d+)+', Number.Integer),
            (r'\d+', Number.Integer)
        ],

        "blockcomment": [
            (r'[^=#]', Comment.Multiline),
            (r'#=', Comment.Multiline, '#push'),
            (r'=#', Comment.Multiline, '#pop'),
            (r'[=#]', Comment.Multiline),
        ],

        'string': [
            (r'"', String, '#pop'),
            # FIXME: This escape pattern is not perfect.
            (r'\\([\\"\'$nrbtfav]|(x|u|U)[a-fA-F0-9]+|\d+)', String.Escape),
            # Interpolation is defined as "$" followed by the shortest full
            # expression, which is something we can't parse.
            # Include the most common cases here: $word, and $(paren'd expr).
            (r'\$' + allowed_variable, String.Interpol),
            # (r'\$[a-zA-Z_]+', String.Interpol),
            (r'(\$)(\()', bygroups(String.Interpol, Punctuation), 'in-intp'),
            # @printf and @sprintf formats
            (r'%[-#0 +]*([0-9]+|[*])?(\.([0-9]+|[*]))?[hlL]?[E-GXc-giorsux%]',
             String.Interpol),
            (r'.|\s', String),
        ],

        'tqstring': [
            (r'"""', String, '#pop'),
            (r'\\([\\"\'$nrbtfav]|(x|u|U)[a-fA-F0-9]+|\d+)', String.Escape),
            (r'\$' + allowed_variable, String.Interpol),
            (r'(\$)(\()', bygroups(String.Interpol, Punctuation), 'in-intp'),
            (r'.|\s', String),
        ],

        'regex': [
            (r'"', String.Regex, '#pop'),
            (r'\\"', String.Regex),
            (r'.|\s', String.Regex),
        ],

        'tqregex': [
            (r'"""', String.Regex, '#pop'),
            (r'.|\s', String.Regex),
        ],

        'command': [
            (r'`', String.Backtick, '#pop'),
            (r'\$' + allowed_variable, String.Interpol),
            (r'(\$)(\()', bygroups(String.Interpol, Punctuation), 'in-intp'),
            (r'.|\s', String.Backtick)
        ],

        'in-intp': [
            (r'\(', Punctuation, '#push'),
            (r'\)', Punctuation, '#pop'),
            include('root'),
        ]
    }

    def analyse_text(text):
        return shebang_matches(text, r'julia')


class JuliaConsoleLexer(Lexer):
    """
    For Julia console sessions. Modeled after MatlabSessionLexer.

    .. versionadded:: 1.6
    """
    name = 'Julia console'
    aliases = ['jlcon']

    def get_tokens_unprocessed(self, text):
        jllexer = JuliaLexer(**self.options)
        start = 0
        curcode = ''
        insertions = []
        output = False
        error = False

        for line in text.splitlines(True):
            if line.startswith('julia>'):
                insertions.append((len(curcode), [(0, Generic.Prompt, line[:6])]))
                curcode += line[6:]
                output = False
                error = False
            elif line.startswith('help?>') or line.startswith('shell>'):
                yield start, Generic.Prompt, line[:6]
                yield start + 6, Text, line[6:]
                output = False
                error = False
            elif line.startswith('      ') and not output:
                insertions.append((len(curcode), [(0, Text, line[:6])]))
                curcode += line[6:]
            else:
                if curcode:
                    for item in do_insertions(
                            insertions, jllexer.get_tokens_unprocessed(curcode)):
                        yield item
                    curcode = ''
                    insertions = []
                if line.startswith('ERROR: ') or error:
                    yield start, Generic.Error, line
                    error = True
                else:
                    yield start, Generic.Output, line
                output = True
            start += len(line)

        if curcode:
            for item in do_insertions(
                    insertions, jllexer.get_tokens_unprocessed(curcode)):
                yield item
