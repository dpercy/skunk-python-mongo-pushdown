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
        COMPARE_OPS = {
            ast.IsNot: '$ne',
            ast.Is: '$eq',
            ast.Gt: '$gt',
            ast.Lt: '$lt',
            ast.GtE: '$gte',
            ast.LtE: '$lte',
        }
        if type(op) in COMPARE_OPS:
            agg_op = COMPARE_OPS[type(op)]
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
    elif isinstance(node, ast.Num):
        return node.n
    elif isinstance(node, ast.Dict):
        if not all(isinstance(k, ast.Str) for k in node.keys):
            raise TypeError('each key in a dict expression must be a string literal')
        if any(k.s.startswith('$') for k in node.keys):
            raise TypeError('keys in a dict expression may not start with "$"')
        return {
            k.s: compile_expr_to_expr(v, env, globals)
            for k, v in zip(node.keys, node.values)
        }
    else:
        raise TypeError('unhandled Expr node type: {}'.format(node))


def compile_function_to_pipeline(func):
    '''
    func: List[BSON] -> List[BSON]
    return: agg pipeline
    '''
    bindings_dict = inspect.getcallargs(func, 'COLLECTION')
    [arg_name] = bindings_dict.keys()
    assert len(func.func_code.co_freevars) == 0, "Can't handle closures yet"
    assert len(func.func_code.co_cellvars) == 0, "Can't handle closures yet"

    statements = code_to_ast(func.func_code).body
    return compile_statements_to_pipeline(statements, arg_name)

def compile_statements_to_pipeline(nodes, collection_variable):
    '''
    collection_variable is the name of the local variable that
    represents the input to the agg pipeline.
    '''
    if len(nodes) == 0:
        # If the Python function returns None, we can't
        # compile it to an agg pipeline, because agg pipelines
        # always return a stream of documents.
        raise TypeError('agg pipeline function must return a value')

    node = nodes[0]
    nodes = nodes[1:]
    if isinstance(node, ast.Return):
        return compile_expr_to_pipeline(node.value, collection_variable, globals)
    else:
        raise TypeError('unhandled Stmt node type: {}'.format(node))

def compile_expr_to_pipeline(node, collection_variable, globals):
    if isinstance(node, ast.Name) and node.id == collection_variable:
        return []
    elif isinstance(node, ast.ListComp):
        # list comprehensions have:
        # - a sequence of "generators", each with
        #     - an "iter" expr
        #     - a sequence of "ifs" exprs
        #     - a "target" for assignment
        # - an "elt" expression

        # Is the comprehension map-and-filter-like?  [ expr for doc in pipeline if condition ... ]
        if len(node.generators) == 1 \
                and isinstance(node.generators[0].target, ast.Name):

            previous_stages = compile_expr_to_pipeline(node.generators[0].iter, collection_variable, globals)
            doc_var = node.generators[0].target.id
            env = { node.generators[0].target.id: "$$CURRENT" }
            filters = [
                {'$filter': {'$expr': compile_expr_to_expr(check, env, globals) } }
                for check in node.generators[0].ifs
            ]
            if isinstance(node.elt, ast.Name) and node.elt.id == doc_var:
                projection = []
            else:
                projection = [
                    {'$replaceRoot': {'newRoot': compile_expr_to_expr(node.elt, env, globals) }}
                ]
            return previous_stages + filters + projection
        else:
            raise ValueError('unhandled list comprehension: {}'.format(node))
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


## TODO support functions that operate on whole collections

thing = compile_function_to_pipeline(lambda docs:
            [ doc for doc in docs
              if doc.x > 3 ])
assert thing == [
        {'$filter': { '$expr':
            {'$gt': [{'$ifNull': ["$$CURRENT.x", None]},
                     {'$ifNull': [3, None]}]}
        }},
    ]

assert compile_function_to_pipeline(lambda docs: [ doc.x for doc in docs ]) == [
    {'$replaceRoot': {'newRoot': "$$CURRENT.x"}},
]
assert compile_function_to_pipeline(lambda docs: [ {'foo': doc.x} for doc in docs ]) == [
    {'$replaceRoot': {'newRoot': {'foo': "$$CURRENT.x"}}},
]

## TODO wrap this up in a nice coll.query(lambda docs: ...) kind of interface


## TODO seek out some really hairy examples (John or Graham's PR; Nathan)
# - but this project shines more when some Python logic is duplicated into a query
# get_num_certified converted to get_num_certified_by_quarter
#   https://github.com/10gen/mitx/commit/502f908ffc#diff-7cb5c0ad755eab9c85042b2fc732e3b4R2724
# get_graded_problems looks interesting
# get_registration_order_and_offering is tricky because of lookup
