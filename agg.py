
import inspect

from parsefunc import (
    reconstruct,
    Global,
    Arg,
    Call,
    GetAttr,
    Const,
    Compare,
)

class CompileFailure(Exception):
    pass

def compile_agg_expr(ast, doc_name):
    if isinstance(ast, Arg):
        assert ast.name == doc_name
        return "$$CURRENT"
    elif isinstance(ast, GetAttr):
        o = compile_agg_expr(ast.object, doc_name)
        if isinstance(o, basestring):
            return o + '.' + ast.name
        else:
            raise CompileFailure("Can't do .{} on {}".format(ast.name, o))
    elif isinstance(ast, Compare):
        if ast.op == '>':
            mongo_op = '$gt'
        else:
            raise CompileFailure("Can't do comparator {}".format(ast.op))
        return { mongo_op: [ compile_agg_expr(ast.left, doc_name),
                             compile_agg_expr(ast.right, doc_name) ] }
    elif isinstance(ast, Const):
        # TODO support other bson...
        if not isinstance(ast.value, (int, float, basestring)):
            raise CompileFailure("Can't do this constant: {}".format(ast.value))
        return {'$literal': ast.value}
    else:
        raise CompileFailure("No case for {}".format(ast))

def agg_expr_func(func):
    body = reconstruct(func)
    args = inspect.getargs(func.func_code)
    assert args.varargs is None
    assert args.keywords is None
    [arg] = args.args
    return compile_agg_expr(body, arg)

assert agg_expr_func(lambda doc: doc.x) == "$$CURRENT.x"
assert agg_expr_func(lambda doc: doc.x > 3) == {'$gt': ["$$CURRENT.x", {'$literal': 3}]}


