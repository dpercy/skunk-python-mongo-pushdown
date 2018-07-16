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

    assert len(func.func_code.co_freevars) == 0, "Can't handle closures yet: {}".format(func)
    assert len(func.func_code.co_cellvars) == 0, "Can't handle closures yet: {}".format(func)

    statements = code_to_ast(func.func_code).body
    env = {
        arg_name: '$$CURRENT',
    }
    globals = func.func_globals
    return compile_statements_to_expr(statements, env, globals)

def compile_statements_to_expr(nodes, env, globals):
    '''
    nodes: List[ast.Node]
    env: Dict[string -> AggExpr]
    globals: Dict[string -> PythonValue]
    '''
    if len(nodes) == 0:
        return {'$literal': None}

    node = nodes[0]
    nodes = nodes[1:]

    if isinstance(node, ast.Return):
        return compile_expr_to_expr(node.value, env, globals)
    elif isinstance(node, ast.If):
        agg_if = compile_expr_to_expr(node.test, env, globals)
        # Append the statements outside the If to both branches of the If.
        # That's a simple way to handle fallthrough.
        agg_then = compile_statements_to_expr(node.body + nodes, env, globals)
        agg_else = compile_statements_to_expr(node.orelse + nodes, env, globals)
        return {'$cond': {
            'if': agg_if,
            'then': agg_then,
            'else': agg_else,
        }}
    else:
        raise TypeError('unhandled Stmt node type: {}'.format(node))


def compile_expr_to_expr(node, env, globals):
    def recur(node):
        return compile_expr_to_expr(node, env, globals)

    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        elif node.id == 'None':
            return None
        else:
            raise ValueError('unhandled global variable {}'.format(node.id))
    elif isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return {'$and': [recur(n) for n in node.values]}
        else:
            raise TypeError('unhandled BoolOp op type: {}'.format(node.op))
    elif isinstance(node, ast.Compare):
        assert len(node.comparators) == 1, "Can't do chained comparisons yet"
        left = node.left
        [op] = node.ops
        [right] = node.comparators
        if isinstance(op, ast.IsNot):
            agg_op = '$ne'
        elif isinstance(op, ast.Is):
            agg_op = '$eq'
        else:
            raise ValueError('unhandled Compare op {}'.format(op))


        # When doing equality checks, consider missing fields and null fields to be equal.
        # This is weird, but similar to what MongoEngine does.
        # Maybe this is incompatible with the PyMODM way of doing things.
        def _convert_missing_null(e):
            if e is None:
                return e
            return {'$ifNull': [e, None]}
        return { agg_op: [ _convert_missing_null(recur(left)),
                           _convert_missing_null(recur(right)) ]}
    elif isinstance(node, ast.Attribute):
        agg_doc = recur(node.value)
        # If we're trying to dereference an agg variable or path,
        # we just append ".fieldname" to it.
        if isinstance(agg_doc, basestring):
            assert agg_doc[0] == '$', 'Field paths must start with "$": {}'.format(agg_doc)
            return agg_doc + '.' + node.attr
        else:
            # If the expression isn't a field path, we have to use $let
            # to bind it to a variable so we can dereference it.
            return {'$let': {'vars': {'x': agg_doc}, 'in': '$$x.' + node.attr}}
    elif isinstance(node, ast.Call):
        assert isinstance(node.func, ast.Name), "Can't handle calls to function expressions yet"
        func_name = node.func.id
        # Try to look up the function
        if func_name in env:
            raise ValueError("Can't handle non-global functions yet")
        elif func_name in globals:
            func_value = globals[func_name]

            # Try to inline the function!
            # - match up arguments with paramters
            assert node.starargs is None, "unhandled starargs"
            assert node.kwargs is None, "unhandled kwargs"
            arg_expr_dict = inspect.getcallargs(func_value, *node.args)
            # - compile each argument
            arg_agg_dict = { name: compile_expr_to_expr(e, env, globals)
                             for name, e in arg_expr_dict.items() }
            # - compile the function body
            body_env = { name: "$$" + name
                         for name in arg_expr_dict }
            body = compile_statements_to_expr(code_to_ast(func_value.func_code).body, body_env, func_value.func_globals)
            # - wrap the result in a $let
            return {'$let': {
                'vars': arg_agg_dict,
                'in': body
            }}
        else:
            raise TypeError('unbound function name (maybe a problem with import order?): {}'.format(func_name))
    else:
        raise TypeError('unhandled Expr node type: {}'.format(node))


##### examples

def is_pending_retirement(doc):
    return (doc.retirement is not None
        and doc.retirement.retired is None)

assert compile_function_to_expr(is_pending_retirement) == \
    {'$and': [ {'$ne': [{'$ifNull': ["$$CURRENT.retirement", None]}, None]},
               {'$eq': [{'$ifNull': ["$$CURRENT.retirement.retired", None]}, None]} ]}


def get_draft_or_public(profile):
    if profile.draft is not None:
        return profile.draft
    else:
        return profile.public

    assert False, 'unreachable'

assert compile_function_to_expr(get_draft_or_public) == \
    {'$cond': {
        'if': {'$ne': [{'$ifNull': ["$$CURRENT.draft", None]}, None]},
        'then': "$$CURRENT.draft",
        'else': "$$CURRENT.public",
    }}

def get_latest_name(profile):
    return get_draft_or_public(profile).first_name

assert compile_function_to_expr(get_latest_name) == \
    {'$let': {
        'vars': {
            'x':
                {'$let': {
                    'vars': {'profile': "$$CURRENT"},
                    'in':
                        {'$cond': {
                            'if': {'$ne': [{'$ifNull': ["$$profile.draft", None]}, None]},
                            'then': "$$profile.draft",
                            'else': "$$profile.public",
                        }}}}
        },
        'in': "$$x.first_name"
    }}


## TODO wrap this up in a nice coll.query(lambda docs: ...) kind of interface


## TODO seek out some really hairy examples (John or Graham's PR; Nathan)
