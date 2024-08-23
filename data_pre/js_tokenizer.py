# -*- coding: utf-8 -*-
"""
@Time ： 2024/8/23 11:08
@Auth ： fangsongtao
@File ：js_tokenizer.py
@IDE ：PyCharm
"""
# coding:utf-8
from tree_sitter import Language, Parser
from utils import *

'''
缩进仅是用于格式化，不影响语义，所以不生成缩进，可以在后处理阶段根据生成代码内容进行格式美化
对有歧义的地方";" 与 '\n' 均可，但标准上是用";"，因此在合适的地方统一插入 ";"
与java类似，注解没有显示的结束符，在注解语句后插入标识符
与java类似 for内";" 与 行末的 ";" 含义不同，需要单独的for内";"标识符
'''


class WordNode(object):
    def __init__(self, node, type, row, col):
        self.node = node
        self.type = type
        self.row = row
        self.col = col

    def get_value(self):
        if self.type == "semicolon" or self.type == 'for_left' or self.type == 'for_right':
            value = -0.8
        elif self.type == "s_method" or self.type == "s_if" or self.type == 's_for' or self.type == 's_while' or self.type == \
                's_try' or self.type == 's_switch':
            value = -0.5
        elif self.type == "e_method" or self.type == "e_if" or self.type == 'e_for' or self.type == 'e_while' or self.type == \
                'e_try' or self.type == 'e_switch':
            value = -0.5
        elif self.type == "annotation":
            value = 1
        else:
            value = 0
        return self.row * 10000000 + self.col * 2 + value


def gen_node(node):
    return WordNode(node, node.type, node.start_point[0], node.start_point[1])


def gen_semicolon_node(node):
    return WordNode("<;>", "semicolon", node.end_point[0], node.end_point[1])


def left_flag_node(node):
    return WordNode("<for_left>", "for_left", node.end_point[0], node.end_point[1])


def right_flag_node(node):
    return WordNode("<for_right>", "for_right", node.end_point[0], node.end_point[1])


def gen_annotation_node(node):
    return WordNode(ANNOTATION_END, "annotation", node.end_point[0], node.end_point[1])


def gen_if_node(choice, node):
    if choice == 0:  # start
        return WordNode(IF_START, "s_if", node.start_point[0], node.start_point[1])
    else:
        return WordNode(IF_END, "e_if", node.end_point[0], node.end_point[1])


def gen_for_node(choice, node):
    if choice == 0:  # start
        return WordNode(FOR_START, "s_for", node.start_point[0], node.start_point[1])
    else:
        return WordNode(FOR_END, "e_for", node.end_point[0], node.end_point[1])


def gen_while_node(choice, node):
    if choice == 0:  # start
        return WordNode(WHILE_START, "s_while", node.start_point[0], node.start_point[1])
    else:
        return WordNode(WHILE_END, "e_while", node.end_point[0], node.end_point[1])


def gen_method_node(choice, node):
    if choice == 0:  # start
        return WordNode(METHOD_START, "s_method", node.start_point[0], node.start_point[1])
    else:
        return WordNode(METHOD_END, "e_method", node.end_point[0], node.end_point[1])


def gen_try_node(choice, node):
    if choice == 0:  # start
        return WordNode(TRY_START, "s_try", node.start_point[0], node.start_point[1])
    else:
        return WordNode(TRY_END, "e_try", node.end_point[0], node.end_point[1])


def gen_switch_node(choice, node):
    if choice == 0:  # start
        return WordNode(SWITCH_START, "s_switch", node.start_point[0], node.start_point[1])
    else:
        return WordNode(SWITCH_END, "e_switch", node.end_point[0], node.end_point[1])


annotation_type = set(["decorator"])
semicolon_type = set(["field_definition",
                      "expression_statement",
                      "lexical_declaration",
                      "variable_declaration",
                      "import_statement",
                      "export_statement",
                      "break_statement",
                      "continue_statement",
                      "debugger_statement",
                      "throw_statement",
                      "return_statement",
                      ])


def get_leaf_nodes(root):
    nodes = [root]
    leaf_nodes = []
    ind = 0
    while ind < len(nodes):
        if len(nodes) > MAX_EXPAND_NODES:
            raise Exception("too many nodes")
        node = nodes[ind]
        flag_for = 0
        if node.type in semicolon_type:

            leaf_nodes.append(gen_semicolon_node(node))
        elif node.type in annotation_type:
            leaf_nodes.append(gen_annotation_node(node))
        if node.type == 'function_declaration':
            leaf_nodes.append(gen_method_node(0, node))
            leaf_nodes.append(gen_method_node(1, node))
        if node.type == 'for_statement':
            leaf_nodes.append(gen_for_node(0, node))
            leaf_nodes.append(gen_for_node(1, node))
        if node.type == 'while_statement':
            leaf_nodes.append(gen_while_node(0, node))
            leaf_nodes.append(gen_while_node(1, node))
        if node.type == 'if_statement':
            leaf_nodes.append(gen_if_node(0, node))
            leaf_nodes.append(gen_if_node(1, node))
        if node.type == 'try_statement':
            leaf_nodes.append(gen_try_node(0, node))
            leaf_nodes.append(gen_try_node(1, node))
        if node.type == 'switch_statement':
            leaf_nodes.append(gen_switch_node(0, node))
            leaf_nodes.append(gen_switch_node(1, node))
        children = node.children
        if len(children) == 0:
            if node.text.decode('utf-8') == '(' or node.text.decode('utf-8') == ')':
                if node.parent.type == 'for_statement' and node.text.decode('utf-8') == '(':
                    leaf_nodes.append(left_flag_node(node))
                    leaf_nodes.append(gen_node(node))
                elif node.parent.type == 'for_statement' and node.text.decode('utf-8') == ')':
                    leaf_nodes.append(right_flag_node(node))
                    leaf_nodes.append(gen_node(node))
                else:
                    leaf_nodes.append(gen_node(node))
            else:
                leaf_nodes.append(gen_node(node))
        else:
            nodes += children
        ind += 1
    return leaf_nodes


P = re.compile("<for_left>(.*?)<for_right>")


def post_process(tokens):
    sen = " ".join(tokens)
    res_sen = ""
    s = 0
    for m in P.finditer(sen):
        res_sen += sen[s:m.start()].replace("<for>", "").replace("<;>", ";")
        res_sen += m.group(1).replace("<;>", "")
        s = m.end()
    res_sen += sen[s:].replace("<for>", "").replace("<;>", ";")
    res_tokens = [x for x in res_sen.split(" ") if len(x) > 0]
    return res_tokens


class JavascriptCodeTokenizer(object):
    def __init__(self):
        language = Language('build/languages.so', 'javascript')
        parser = Parser()
        parser.set_language(language)
        self.parser = parser
        self.comment_type = set(['comment'])
        self.string_type = set(['string', 'template_string'])

    def is_comment(self, node):
        if node.type in self.comment_type:
            return True
        if node.type in self.string_type and node.node.prev_sibling is None and node.node.next_sibling is None:
            return True
        return False

    def tokenize(self, code):
        tree = self.parser.parse(bytes(code, "utf-8"))
        nodes = get_leaf_nodes(tree.root_node)
        nodes.sort(key=lambda x: x.get_value())
        tokens = []
        ind = 0
        while ind < len(nodes):
            node = nodes[ind]
            # 新行
            if node.type == "annotation":
                tokens.append(ANNOTATION_END)
            elif node.type == "semicolon":
                if tokens[-1] != ";" and tokens[-1] != "<;>":
                    tokens.append("<;>")
            elif node.type == "for_left":
                tokens.append("<for_left>")
            elif node.type == "for_right":
                tokens.append("<for_right>")
            # 注释
            elif self.is_comment(node):
                tokens.append(COMMENT_START)
                tokens.extend(process_text(node.node.text, "string"))
                tokens.append(COMMENT_END)
            elif node.type == "s_method":
                # print('method done')
                tokens.append(METHOD_START)
            elif node.type == 'e_method':
                # print('/me done')
                tokens.append(METHOD_END)
            elif node.type == 's_if':
                # print('if done')
                tokens.append(IF_START)
            elif node.type == 'e_if':
                # print('/if done')
                tokens.append(IF_END)
            elif node.type == 's_while':
                tokens.append(WHILE_START)
            elif node.type == 'e_while':
                tokens.append(WHILE_END)
            elif node.type == 's_for':
                tokens.append(FOR_START)
            elif node.type == 'e_for':
                tokens.append(FOR_END)
            elif node.type == 's_try':
                tokens.append(TRY_START)
            elif node.type == 'e_try':
                tokens.append(TRY_END)
            elif node.type == 's_switch':
                tokens.append(SWITCH_START)
            elif node.type == 'e_switch':
                tokens.append(SWITCH_END)
            else:
                # 节点文本
                if node.type in self.string_type:
                    tokens.extend(process_text(node.node.text, "string"))
                else:
                    if is_for_semicolon(node.node):
                        tokens.append(FOR_SEMICOLON)
                    else:
                        tokens.extend(process_text(node.node.text))
            ind += 1
        return post_process(tokens)


def test(file_path, save_path):
    import json
    import logging
    tokenizer = JavascriptCodeTokenizer()
    with open(save_path, 'w', encoding='utf-8') as fw:
        with open(file_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip()
                if len(line) > 0:
                    tmp = json.loads(line)
                    try:
                        tokens = tokenizer.tokenize(tmp['content'])
                        tmp['tokens'] = ' '.join(tokens)
                        fw.write(json.dumps(tmp, ensure_ascii=False))
                        fw.write('\n')
                    except Exception as e:
                        logging.exception(e)


def test(file_path, save_path):
    import json
    import logging
    tokenizer = JavascriptCodeTokenizer()
    with open(save_path, 'w', encoding='utf-8') as fw:
        with open(file_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip()
                if len(line) > 0:
                    tmp = json.loads(line)
                    try:
                        tokens = tokenizer.tokenize(tmp['content'])
                        tmp['tokens'] = ' '.join(tokens)
                        fw.write(json.dumps(tmp, ensure_ascii=False))
                        fw.write('\n')
                    except Exception as e:
                        logging.exception(e)


if __name__ == '__main__':
    import sys

    # test('js_node.txt', 'leaf_test.txt')
    test('js_test', 'js_100.txt')
    # test(sys.argv[1], sys.argv[2])
