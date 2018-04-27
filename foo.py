import meta, ast, dis

# TODO see http://hackflow.com/blog/2015/03/29/metaprogramming-beyond-decency/

f = lambda doc: doc.x == 1 and doc.y == 2
dis.dis(f.func_code)
f_tree = meta.decompile(f.func_code)

def g(doc):
    doc.x == 1 and doc.y == 2
    # note we ignore the return value here
dis.dis(g.func_code)
g_tree = meta.decompile(g.func_code)
