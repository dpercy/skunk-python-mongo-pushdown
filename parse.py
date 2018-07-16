import uncompyle6
from cStringIO import StringIO

class DummyFD(object):
    def write(self, value):
        pass

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


