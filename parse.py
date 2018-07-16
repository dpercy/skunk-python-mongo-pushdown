import ast
from cStringIO import StringIO
import inspect

import uncompyle6

def code_to_string(code):
    fd = StringIO()
    try:
        uncompyle6.code_deparse(code, out=fd)
        return fd.getvalue()
    finally:
        fd.close()

def code_to_ast(code):
    text = code_to_string(code)
    return ast.parse(text)



'''

agg expression: returns a bson

agg pipeline stage: stream<doc> -> stream<doc>

'''

def compile_function_to_expr(func):
    '''
    func: BSON -> BSON
    return: agg expression that mentions $$CURRENT
    '''
    # check the function arity:
    # if we passed in one positional argument, what would it be named?
    bindings_dict = inspect.getcallargs(func, 'DOCUMENT')
    [arg_name] = bindings_dict.keys()

    statements = code_to_ast(func.func_code).body
    return compile_statements_to_expr(statements, {
        arg_name: '$$CURRENT',
    })

def compile_statements_to_expr(nodes, env):
    '''
    nodes: List[ast.Node]
    env: Dict[string -> AggExpr]
    '''
    if len(nodes) == 0:
        return {'$literal': None}

    node = nodes[0]
    nodes = nodes[1:]

    if isinstance(node, ast.Return):
        return compile_expr_to_expr(node.value, env)
    elif isinstance(node, ast.If):
        agg_if = compile_expr_to_expr(node.test, env)
        # Append the statements outside the If to both branches of the If.
        # That's a simple way to handle fallthrough.
        agg_then = compile_statements_to_expr(node.body + nodes, env)
        agg_else = compile_statements_to_expr(node.orelse + nodes, env)
        return {'$cond': {
            '$if': agg_if,
            '$then': agg_then,
            '$else': agg_else,
        }}
    else:
        raise TypeError('unhandled Stmt node type: {}'.format(node))


def compile_expr_to_expr(node, env):
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        elif node.id == 'None':
            return None
        else:
            raise ValueError('unhandled global variable {}'.format(node.id))
    elif isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return {'$and': [compile_expr_to_expr(n, env) for n in node.values]}
        else:
            raise TypeError('unhandled BoolOp op type: {}'.format(node.op))
    elif isinstance(node, ast.Compare):
        assert len(node.comparators) == 1, "Can't do chained comparisons yet"
        left = node.left
        [op] = node.ops
        [right] = node.comparators
        if isinstance(op, ast.IsNot):
            agg_op = '$neq'
        elif isinstance(op, ast.Is):
            agg_op = '$eq'
        else:
            raise ValueError('unhandled Compare op {}'.format(op))
        return { agg_op: [ compile_expr_to_expr(left, env),
                           compile_expr_to_expr(right, env) ]}
    elif isinstance(node, ast.Attribute):
        agg_doc = compile_expr_to_expr(node.value, env)
        # If we're trying to dereference an agg variable or path,
        # we just append ".fieldname" to it.
        if isinstance(agg_doc, basestring):
            assert agg_doc[0] == '$', 'Field paths must start with "$": {}'.format(agg_doc)
            return agg_doc + '.' + node.attr
        else:
            # Otherwise, we have to do something else.
            # Maybe use $let to bind the expression to a variable.
            raise TypeError("Can't dereference this thing; it's not a field-path: {}".format(agg_doc))
    else:
        raise TypeError('unhandled Expr node type: {}'.format(node))


##### examples

def is_pending_retirement(doc):
    return (doc.retirement is not None
        and doc.retirement.retired is None)

assert compile_function_to_expr(is_pending_retirement) == \
    {'$and': [ {'$neq': ["$$CURRENT.retirement", None]},
               {'$eq':  ["$$CURRENT.retirement.retired", None]} ]}


def get_draft_or_public(profile):
    if profile.draft is not None:
        return profile.draft
    else:
        return profile.public

    assert False, 'unreachable'

assert compile_function_to_expr(get_draft_or_public) == \
    {'$cond': {
        '$if': {'$neq': ["$$CURRENT.draft", None]},
        '$then': "$$CURRENT.draft",
        '$else': "$$CURRENT.public",
    }}
