
import dis
from collections import namedtuple
import expr

# struct view of instructions
LoadGlobal = namedtuple('LoadGlobal', 'name')
LoadFast   = namedtuple('LoadFast', 'index')
LoadAttr   = namedtuple('LoadAttr', 'name')
CallFunction = namedtuple('CallFunction', 'arity')
ReturnValue = namedtuple('ReturnValue', '')
LoadConst  = namedtuple('LoadConst', 'value')
Binop  = namedtuple('Binop', 'op')

class UnrecognizedInstruction(Exception):
    pass

class Scanner(object):
    def __init__(self, bytecodes):
        self.bytecodes = bytecodes
        self.idx = 0

    def done(self):
        return self.idx >= len(self.bytecodes)

    def get_byte(self):
        c = self.bytecodes[self.idx]
        self.idx += 1
        return ord(c)

    def get_short(self):
        lo = self.bytecodes[self.idx]
        hi = self.bytecodes[self.idx+1]
        self.idx += 2
        return (ord(hi) << 8) | ord(lo)


def tokenize(code):
    scanner = Scanner(code.co_code)
    while not scanner.done():
        opcode = scanner.get_byte()
        opname = dis.opname[opcode]
        if opname == 'LOAD_GLOBAL':
            arg = scanner.get_short()
            yield LoadGlobal(code.co_names[arg])
        elif opname == 'LOAD_ATTR':
            arg = scanner.get_short()
            yield LoadAttr(code.co_names[arg])
        elif opname == 'LOAD_FAST':
            arg = scanner.get_short()
            yield LoadFast(arg)
        elif opname == 'CALL_FUNCTION':
            arg = scanner.get_short()
            yield CallFunction(arg)
        elif opname == 'RETURN_VALUE':
            yield ReturnValue()
        elif opname == 'LOAD_CONST':
            arg = scanner.get_short()
            yield LoadConst(code.co_consts[arg])
        elif opname == 'COMPARE_OP':
            arg = scanner.get_short()
            yield Binop(dis.cmp_op[arg])
        elif opname == 'BINARY_AND':
            yield Binop('&')
        elif opname == 'DUP_TOP':
            yield Dup()
        else:
            raise UnrecognizedInstruction(opname)


example1 = lambda x: f(g(x))
assert list(tokenize(example1.func_code)) == [
    LoadGlobal(name='f'),
    LoadGlobal(name='g'),
    LoadFast(index=0),
    CallFunction(arity=1),
    CallFunction(arity=1),
    ReturnValue()
]

example2 = lambda x: x.r
assert list(tokenize(example2.func_code)) == [
    LoadFast(index=0),
    LoadAttr(name='r'),
    ReturnValue()
]

def _pop_n(stack, num):
    values = stack[-num:]
    stack[-num:] = []
    return values

def reconstruct(func):
    code = func.func_code

    # list where each index is an AST representing that local or argument
    local_vars = []
    for idx, name in enumerate(code.co_varnames):
        if idx < code.co_argcount:
            local_vars.append(expr.Arg(name))
        else:
            assert False, "TODO implement local variables?"

    stack = [] # of ASTs
    for token in tokenize(code):
        if isinstance(token, LoadGlobal):
            stack.append(expr.Global(token.name))
        elif isinstance(token, LoadFast):
            stack.append(local_vars[token.index])
        elif isinstance(token, LoadAttr):
            obj = stack.pop()
            stack.append(expr.GetAttr(obj, token.name))
        elif isinstance(token, CallFunction):
            args = _pop_n(stack, token.arity)
            func = stack.pop()
            stack.append(expr.Call(func, args))
        elif isinstance(token, ReturnValue):
            return stack.pop()
        elif isinstance(token, LoadConst):
            stack.append(expr.Const(token.value))
        elif isinstance(token, Binop):
            left, right = _pop_n(stack, 2)
            stack.append(expr.Binop(token.op, left, right))
        else:
            assert False, "Unhandled token type: {}".format(token)




assert reconstruct(example1) == expr.Call(expr.Global('f'), [expr.Call(expr.Global('g'), [expr.Arg('x')])])
assert reconstruct(example2) == expr.GetAttr(expr.Arg('x'), 'r')
assert reconstruct(lambda: a & b) == expr.Binop('&', expr.Global('a'), expr.Global('b'))
