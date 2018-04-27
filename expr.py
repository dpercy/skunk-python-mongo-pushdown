from collections import namedtuple

Global = namedtuple('Global', 'name')
Arg = namedtuple('Arg', 'name')
Call = namedtuple('Call', 'func args')
GetAttr = namedtuple('GetAttr', 'object name')
Const = namedtuple('Const', 'value')
Compare = namedtuple('Compare', 'op left right')
