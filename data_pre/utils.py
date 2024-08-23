# -*- coding: utf-8 -*-
"""
@Time ： 2024/8/23 11:09
@Auth ： fangsongtao
@File ：util.py
@IDE ：PyCharm
"""
# coding:utf-8
import re
from tree_sitter import Language

WORD = re.compile("([_0-9a-zA-Z\\\\]+)|(//)|(/\*)|(\*/)|(\\\n)|(\'\'\')|(\"\"\")")

LINE_END = '</line>'
COMMENT_START = '<comment>'
COMMENT_END = '</comment>'
INDENT = '<indent>'
DEDENT = '<dedent>'
FOR_SEMICOLON = "<for>"
ANNOTATION_END = "</annotation>"

METHOD_START = '<bs_Method>'
METHOD_END = '<be_Method>'
TRY_START = '<bs_try>'
TRY_END = '<be_try>'
SWITCH_START = '<bs_switch>'
SWITCH_END = '<be_switch>'
IF_START = '<bs_If>'
IF_END = '<be_If>'
FOR_START = '<bs_For>'
FOR_END = '<be_For>'
WHILE_START = '<bs_While>'
WHILE_END = '<be_While>'

MAX_EXPAND_NODES = 100000

flag = Language.build_library(
    'build/languages.so',

    # Include one or more languages
    [
        'tree-sitter-java',
        'tree-sitter-javascript',
        'tree-sitter-python',
        'tree-sitter-bash',
        'tree-sitter-c',
        'tree-sitter-cpp',
        'tree-sitter-c-sharp',
        'tree-sitter-go',
        'tree-sitter-php',
        'tree-sitter-ruby',
        'tree-sitter-swift',
        'tree-sitter-typescript/typescript',
        'tree-sitter-typescript/tsx',
        'tree-sitter-rust',
    ]
)

SPACE = set([' ', '\t', '\n'])


def string_token(s):
    tokens = []
    start = 0
    for m in WORD.finditer(s):
        for c in s[start:m.start()]:
            tokens.append(c)
        tokens.append(m.group())
        start = m.end()
    for c in s[start:]:
        if c in SPACE:
            continue
        tokens.append(c)
    return tokens


def process_text(text, typ=""):
    res = []
    text = text.decode('utf-8')
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue
        if typ == "string":
            res.extend(string_token(line))
        else:
            res.append(line)
    return res


def is_for_semicolon(node):
    text = node.text.decode('utf-8')
    if text != ';':
        return False
    if node.parent and node.parent.type == 'for_statement':
        return True
    if node.parent and node.parent.parent and node.parent.parent.type == 'for_statement':
        return True
    return False


DIGITS = re.compile("(0[xX][0-9a-fA-F]+[,;uU ]?)|[0-9]+[,;]?")
STRS_SPLIT = re.compile("[^0-9a-zA-Z_]+")
CHARS = re.compile("[_a-zA-Z]+")
BIG_IDENT = re.compile("(\t{8,})|( {32, })")
OX = re.compile("0[xX]")
LINE = re.compile("[;#=]")


def is_filt(s, language='c'):
    if language.lower() == 'c':
        # 普通语句占比过低
        lines = [x for x in s.split('\n') if len(x.strip()) > 0]
        n = 0
        for line in lines:
            if len(line) > 1024:
                return True
            if LINE.search(line):
                n += 1
        if 1.0 * n / len(lines) < 0.1:
            return True

    # 不包含大量0x数据
    n = 0
    for m in OX.finditer(s):
        n += 1
    if n > 10:
        return True

    # 最大层级不超过8
    if BIG_IDENT.search(s):
        return True

    # 最长连续字符串长度不超过512
    for t in STRS_SPLIT.split(s):
        if len(t) > 512:
            return True
    # 数字占比不超过80%
    n = 0
    for m in DIGITS.finditer(s):
        # print(m.group())
        n += len(m.group())
    # print(1.0 * n / len(s))
    if 1.0 * n / len(s) > 0.5:
        return True
    # 字符占比不低于30%
    n = 0
    for m in CHARS.finditer(s):
        n += len(m.group())
    if 1.0 * n / len(s) < 0.3:
        return True

    return False
