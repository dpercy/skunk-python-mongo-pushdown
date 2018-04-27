import inspect

import expr
from parsefunc import reconstruct


class CompileError(Exception):
    pass


def compile_matcher(func):
    body = reconstruct(func)
    [arg] = inspect.getargs(func.func_code).args
    globals = inspect.getmodule(func)
    return _compile_matcher(body, arg, globals)

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

def _compile_matcher(e, arg, globals):
    if isinstance(e, expr.Binop) and e.op == '&':
        return {
            '$and': [
                _compile_matcher(e.left, arg, globals),
                _compile_matcher(e.right, arg, globals),
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
    elif isinstance(e, expr.Call):
        e = inline_call(e, globals)
        return _compile_matcher(e, arg, globals)
    else:
        raise CompileError

def inline_call(e, globals):
    func = lookup_function(e.func, globals)

    # dirty unsafe unhygienic hack
    return reconstruct(func, e.args)

def lookup_function(e, globals):
    if isinstance(e, expr.Global):
        return getattr(globals, e.name)
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

def tall_enough(doc):
    return doc.height >= 5.2

def young_enough(doc):
    return doc.age < 99

def can_ride_rollercoaster(doc):
    return tall_enough(doc) & young_enough(doc)

assert compile_matcher(can_ride_rollercoaster) == { '$and': [ {'height': {'$gte': 5.2}}, {'age': {'$lt': 99}} ] }
