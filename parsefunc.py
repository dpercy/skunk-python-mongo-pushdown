
import dis
from collections import namedtuple

# struct view of instructions
LoadGlobal = namedtuple('LoadGlobal', 'index')
LoadFast   = namedtuple('LoadFast', 'index')
CallFunction = namedtuple('CallFunction', 'arity')
ReturnValue = namedtuple('ReturnValue', '')

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
            yield LoadGlobal(arg)
        elif opname == 'LOAD_FAST':
            arg = scanner.get_short()
            yield LoadFast(arg)
        elif opname == 'CALL_FUNCTION':
            arg = scanner.get_short()
            yield CallFunction(arg)
        elif opname == 'RETURN_VALUE':
            yield ReturnValue()
        else:
            raise UnrecognizedInstruction(opname)


example1 = lambda x: f(g(x))
assert list(tokenize(example1.func_code)) == [
    LoadGlobal(index=0),
    LoadGlobal(index=1),
    LoadFast(index=0),
    CallFunction(arity=1),
    CallFunction(arity=1),
    ReturnValue()
]

# ASTs
Global = namedtuple('Global', 'name')
Arg = namedtuple('Arg', 'name')
Call = namedtuple('Call', 'func args')

def _pop_n(stack, num):
    values = stack[-num:]
    stack[-num:] = []
    return values

def reconstruct(func):
    code = func.func_code
    global_names = code.co_names
    local_names = code.co_varnames

    # list where each index is an AST representing that local or argument
    local_vars = []
    for idx, name in enumerate(local_names):
        if idx < code.co_argcount:
            local_vars.append(Arg(name))
        else:
            assert False, "TODO implement local variables?"

    stack = [] # of ASTs
    for token in tokenize(code):
        if isinstance(token, LoadGlobal):
            stack.append(Global(global_names[token.index]))
        elif isinstance(token, LoadFast):
            stack.append(local_vars[token.index])
        elif isinstance(token, CallFunction):
            args = _pop_n(stack, token.arity)
            func = stack.pop()
            stack.append(Call(func, args))
        elif isinstance(token, ReturnValue):
            return stack.pop()
        else:
            assert False, "Unhandled token type: {}".format(token)




assert reconstruct(example1) == Call(Global('f'), [Call(Global('g'), [Arg('x')])])
