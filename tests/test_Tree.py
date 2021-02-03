from check_assert import check_assert
from gp.Tree import Tree

import numpy as np


def test_apply_rules_to_tree():

    t = Tree(tree=['*', ['#x'], ['#f']],
             rng=np.random.RandomState(0)) # this makes #f consistent
    t.apply_rules_to_tree()
    yield check_assert, t.get_lisp_string() == '(* (x0) (4.881350392732472))'


def test_apply_rules_to_node():

    t = Tree(tree=None, rng=np.random.RandomState(0))
    for node in ['*', '%', '-', '+', 'sin', 'cos', '10']:
        rule_node = t.apply_rules_to_node(node)
        yield check_assert, rule_node == node

    rule_node = t.apply_rules_to_node('#i')
    yield check_assert, type(rule_node) == int

    rule_node = t.apply_rules_to_node('#f')
    yield check_assert, type(rule_node) == float

    rule_node = t.apply_rules_to_node('#x')
    yield check_assert, rule_node[0] == 'x' and np.all([s.isdigit() for s in rule_node[1:]])


def test_get_lisp_string():

    tree = Tree('(* (3) (1))')
    yield check_assert, tree.get_lisp_string() == '(* (3) (1))'
    yield check_assert, tree.get_lisp_string(actual_lisp=True) == '(* 3 1)'


def test__convert_lisp_to_standard_terminal():

    stack = ['*']

    tree = Tree(tree=None)
    standard = tree._convert_lisp_to_standard_terminal(word='[x0]',
                                                       stack=stack)
    yield check_assert, 'x[0]' == standard

    standard = tree._convert_lisp_to_standard_terminal(word='[0]',
                                                       stack=stack)
    yield check_assert, '0' == standard

    stack = ['*', '*']

    standard = tree._convert_lisp_to_standard_terminal(word='[x0]',
                                                       stack=stack)
    yield check_assert, 'x[0],' == standard

    standard = tree._convert_lisp_to_standard_terminal(word='[0]',
                                                       stack=stack)
    yield check_assert, '0,' == standard


def test__convert_lisp_to_standard_primitive():

    tree = Tree(tree=None)
    stack = []
    standard = tree._convert_lisp_to_standard_primitive(word='[*',
                                                        stack=stack)
    yield check_assert, 'np.multiply(' == standard
    yield check_assert, stack == ['np.multiply']

    stack = []
    standard = tree._convert_lisp_to_standard_primitive(word='[-',
                                                        stack=stack)
    yield check_assert, 'np.subtract(' == standard
    yield check_assert, stack == ['np.subtract']

    # Don't reset stack this time so we get two primitives
    # in the stack.
    standard = tree._convert_lisp_to_standard_primitive(word='[%',
                                                        stack=stack)
    yield check_assert, 'protected_division(' == standard
    yield check_assert, stack == ['np.subtract', 'protected_division']


def test_convert_lisp_to_standard():
    tree = Tree('(* (3) (1))')
    yield check_assert, tree.convert_lisp_to_standard() == 'np.multiply(3,1)+0*x[0]'

    yield check_assert, tree.convert_lisp_to_standard(convertion_dict={'*': 'Apple'}) == 'Apple(3,1)+0*x[0]'


def test_from_string():

    tree = Tree(tree=None, num_vars=2)
    tree.from_string('(% (3) (x0))', actual_lisp=False)
    yield check_assert, tree.get_lisp_string(actual_lisp=True) == '(% 3 x0)'

    tree.from_string('(+ (x0) (1))')
    yield check_assert, tree.get_lisp_string() == '(+ (x0) (1))'

    tree.from_string('(+ x0 (* x1 3))', actual_lisp=True)
    yield check_assert, tree.get_lisp_string(actual_lisp=True) == '(+ x0 (* x1 3))'


def test_select_subtree():
    # Test that select_subtree works if
    # child_index_list is specified by list
    tree = Tree('(* (x0) (* (x1) (x2)))', actual_lisp=False, num_vars=3)
    subtree = tree.select_subtree(child_indices=[1, 0])
    yield check_assert, subtree == ['x1']

    # Test that select_subtree works if
    # child_index_list is specified by tuple
    tree = Tree('(* (x0) (* (x1) (x2)))', actual_lisp=False, num_vars=3)
    subtree = tree.select_subtree(child_indices=(1, 0))
    yield check_assert, subtree == ['x1']


def test_set_subtree_list():
    # Test that set_subtree works if
    # child_index_list is specified by list
    tree = Tree('(* (x0) (* (x0) (x0)))', actual_lisp=False)
    tree2 = Tree('(- (x0) (3))', actual_lisp=False)
    tree.set_subtree(new_subtree=tree2.tree,
                     child_indices=[1, 0])
    yield check_assert, tree.get_lisp_string() == '(* (x0) (* (- (x0) (3)) (x0)))'

    # Test that set_subtree works if
    # child_index_list is specified by tuple
    tree = Tree('(* (x0) (* (x0) (x0)))', actual_lisp=False)
    tree2 = Tree('(- (x0) (3))', actual_lisp=False)
    tree.set_subtree(new_subtree=tree2.tree,
                     child_indices=(1, 0))
    yield check_assert, tree.get_lisp_string() == '(* (x0) (* (- (x0) (3)) (x0)))'


def test_get_node_list():
    tree = Tree('(- (3) (+ (x0) (3)))')
    yield check_assert, tree.get_node_list() == [(), (0,), (1,), (1, 0), (1, 1)]


def test_get_num_leaves():
    tree = Tree('(- (3) (+ (x0) (3)))')
    num_leaves, num_nodes = tree.get_num_leaves()
    yield check_assert, num_leaves == 3
    yield check_assert, num_nodes == 5


def test_get_node_dict():
    tree = Tree('(- (3) (+ (x0) (3)))')
    yield check_assert, tree.get_node_dict() == {(1, 0): 'x0', (): '-', (1,): '+', (0,): 3, (1, 1): 3}


def test_get_node_map():
    tree = Tree('(- (3) (+ (x0) (3)))')
    yield check_assert, tree.get_node_map() == {'-': {()}, 3: {(0,), (1, 1)}, 'x0': {(1, 0)}, '+': {(1,)}}


def test_get_tree_size():

    tree = Tree('(x0)')
    yield check_assert, tree.get_tree_size() == 1

    tree = Tree(tree='(% (3) (x0))', actual_lisp=False)
    assert tree.get_tree_size() == 3


def test_get_depth():
    tree = Tree('(x0)')
    yield check_assert, tree.get_depth() == 0

    tree = Tree('(- (3) (+ (x0) (3)))')
    yield check_assert, tree.get_depth() == 2
