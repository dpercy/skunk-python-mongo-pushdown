import inspect

import expr
from parsefunc import reconstruct


class CompileError(Exception):
    pass


def compile_matcher(func):
    body = reconstruct(func)
    [arg] = inspect.getargs(func.func_code).args
    return _compile_matcher(body, arg)

_match_ops = {
    '<': '$lt',
    '>': '$gt',
    '<=': '$lte',
    '>=': '$gte',
    '==': '$eq',
    '!=': '$ne',
}

_flip_op = {
    '<': '>',
    '>': '<',
    '<=': '>=',
    '>=': '<=',
    '==': '==',
    '!=': '!=',
}

def flip_binop(e):
    return expr.Binop(_flip_op[e.op], e.right, e.left)

def _compile_matcher(e, arg):
    if isinstance(e, expr.Binop) and e.op == '&':
        return {
            '$and': [
                _compile_matcher(e.left, arg),
                _compile_matcher(e.right, arg),
            ]
        }
    elif isinstance(e, expr.Binop) and e.op in _match_ops:
        try:
            path = _compile_path(e.left, arg)
            const = _compile_constant(e.right)
        except CompileError:
            e = flip_binop(e)
            path = _compile_path(e.left, arg)
            const = _compile_constant(e.right)
        op = _match_ops[e.op]
        return { '.'.join(path): { op: const } }
    else:
        raise CompileError

def _compile_constant(e):
    if isinstance(e, expr.Const):
        return e.value
    raise CompileError

def _compile_path(e, arg):
    if isinstance(e, expr.Arg) and e.name == arg:
        return []
    elif isinstance(e, expr.GetAttr):
        p = _compile_path(e.object, arg)
        p.append(e.name)
        return p
    raise CompileError


assert compile_matcher(lambda doc: doc.x > 3) == {'x': {'$gt': 3}}
assert compile_matcher(lambda doc: 1 != doc.x) == {'x': {'$ne': 1}}

# for conjuctions, we don't need to simplify it all the way.
# TODO support the fancier `and` and `or` operators
assert compile_matcher(lambda doc: (doc.x == 1) & (doc.y == 2)) == {'$and': [{'x': {'$eq': 1}}, {'y': {'$eq': 2}}]}


