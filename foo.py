
import dis
from collections import namedtuple

# struct view of instructions
LoadGlobal = namedtuple('LoadGlobal', 'name')
LoadFast   = namedtuple('LoadFast', 'name')  # for arguments
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
    global_names = code.co_names
    local_names = code.co_varnames

    scanner = Scanner(code.co_code)

    while not scanner.done():
        opcode = scanner.get_byte()
        opname = dis.opname[opcode]
        if opname == 'LOAD_GLOBAL':
            arg = scanner.get_short()
            yield LoadGlobal(global_names[arg])
        elif opname == 'LOAD_FAST':
            arg = scanner.get_short()
            yield LoadFast(local_names[arg])
        elif opname == 'CALL_FUNCTION':
            arg = scanner.get_short()
            yield CallFunction(arg)
        elif opname == 'RETURN_VALUE':
            yield ReturnValue()
        else:
            raise UnrecognizedInstruction(opname)


example1 = lambda x: f(g(x))
assert list(tokenize(example1.func_code)) == [
    LoadGlobal(name='f'),
    LoadGlobal(name='g'),
    LoadFast(name='x'),
    CallFunction(arity=1),
    CallFunction(arity=1),
    ReturnValue()
]

