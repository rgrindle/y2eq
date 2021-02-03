"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Oct 9, 2020

PURPOSE: The class Tree is the basis for representing
         an equation as a tree. This will be expanded on
         by the Individual class. Then, the Individual class
         will be used by the GeneticProgramming class.

NOTES:

TODO:
"""
from gp.primitive_info import primitive2function
from gp.exceptions import UnknownPrimitiveError

import pygraphviz as pgv

import collections
from typing import List, Tuple, Dict, Union, Set


class Tree:
    """This class represents a tree as a list of lists.
    Each sublist specifies the child nodes of the current node."""

    def __init__(self, tree,
                 num_vars: int = 1, rng=None,
                 actual_lisp: bool = False, **params):
        """Initialize tree

        Parameters
        ----------
        tree : list (of lists) or str
            If list, this will become the tree. If str, the function
            Tree.from_string will be called to convert the assumed
            lisp to a list of lists.
        num_vars : int (default=1)
            The number of input variables that the tree accepts. All
            input variables are of the form x0, x1, ...
        rng : Random number generator (optional)
            This allows initial seed to carry through layers of class
            and get reproducible runs. Example: rng=np.random.RandomState(0).
            If not specified, running certain methods will not be possible.
        params : key word arguments
            Stuff for child classes.

        Examples
        --------
        >>> t = Tree(tree=['*', ['#x'], ['#f']], rng=np.random.RandomState(0))
        >>> t.get_lisp_string()
        '(* (x0) (4.881350392732472))'

        >>> t = Tree(tree='(% (1) (x0))')
        >>> t.get_lisp_string()
        '(% (1) (x0))'
        """

        self.rng = rng
        self.num_vars = num_vars
        self.params = params

        if type(tree) == list:
            self.tree = tree
            self.apply_rules_to_tree()

        elif type(tree) == str:
            self.from_string(tree, actual_lisp=actual_lisp)
            self.apply_rules_to_tree()

        elif tree is None:
            self.tree = tree

        else:
            print('ERROR in Tree.__init__: Unkonwn type for arugment tree')
            print('Type is ', type(tree))
            exit()

    def apply_rules_to_tree(self, tree=None) -> None:
        """Check each node name in tree (self) to see
        if the name needs adjustment. This could be a
        constant is specified but not given a value
        or a variable is specified but not which variable.

        Parameters
        ----------
        tree : list (of lists, default=self.tree)
            A subtree of self.tree may be specified.
            Tree contains all the nodes to which the
            rules will be applied.

        Examples
        --------
        >>> t = Tree(tree=['*', ['#x'], ['#f']], rng=np.random.RandomState(0))
        >>> t.get_lisp_string()
        '(* (#x) (#f))'

        >>> t.apply_rules_to_tree()
        >>> t.get_lisp_string()
        '(* (x0) (4.881350392732472))'
        """

        tree = self.tree if tree is None else tree

        if type(tree) == list:
            tree[0] = self.apply_rules_to_node(node=tree[0])
            for t in tree[1:]:
                self.apply_rules_to_tree(tree=t)

        else:

            tree = self.apply_rules_to_node(node=tree)

    def apply_rules_to_node(self, node: str) -> Union[int,str]:
        """Here is the actual list of rules. The
        rules are technically only applied to the
        nodes. In the previous function the rules
        are applied to each node (thus applied to
        the tree).

        Rules
        #i becomes an integer
        #f becomes a float
        #x becomes a variable named x0, x1, ..., x9

        Parameters
        ----------
        node : str
        """

        if node == "#i":
            return self.rng.randint(-50, 50)

        elif node == "#f":
            return self.rng.uniform(-50, 50)

        elif node == "#x":
            return 'x' + str(self.rng.choice(self.num_vars))

        else:
            return node

# ------------------------------------------------------------ #
#                   Get Tree as String
# ------------------------------------------------------------ #

    def get_lisp_string(self, subtree=None,
                        actual_lisp: bool = False) -> str:
        """Get string (lisp) representing the tree. Since every
        element of self.tree is a list, it is easiest to write
        the lisp with terminals surround by parenthesis like:
        (* (3) (1)) instead of (* 3 1)

        Parameters
        ----------
        subtree : list (of lists, default=self.tree)
            Can use this argument to get lisp of a
            subtree rather than the entire tree.
        actual_lisp : bool (default=False)
            If True, get the actual lisp. That means
            no parenthesis around terminal nodes.

        Returns
        -------
        lisp : str
            The lisp string with the mentioned twist.

        Examples
        --------
        >>> tree = Tree('(* (3) (1))')
        >>> tree.get_lisp_string()
        (* (3) (1))

        >>> tree.get_lisp_string(actual_lisp=True)
        (* 3 1)
        """

        if subtree is None:
            lisp = str(self.tree).replace('[', '(').replace(']', ')').replace(',', '').replace('\'', '')

        else:
            lisp = str(subtree).replace('[', '(').replace(']', ')').replace(',', '').replace('\'', '')

        if actual_lisp:

            # remove () around terminals
            fixed_lisp_list = []

            for word in lisp.split(' '):

                if '(' in word and ')' in word:
                    fixed_lisp_list.append(word[1:-1])

                else:
                    fixed_lisp_list.append(word)

            lisp = ' '.join(fixed_lisp_list)

        return lisp

    def _convert_lisp_to_standard_terminal(self, word: str,
                                           stack: List[str]) -> str:
        """A portion of convert_lisp_to_standard, which deals
        with the terminals only.

        Parameters
        ----------
        word : str
            The terminal, possibly with extra bits. For example,
            word could be '[x0]'
        stack : list
            The stack of primitives encountered in the list so far.

        Returns
        -------
        standard : str
            The string word converted to standard, potentially
            containing some ) and/or a single ,
        """

        count = word.count(']')

        # if variable
        if word[1] == 'x':
            assert word[-count - 1].isdigit()
            standard = 'x[' + word[-count - 1] + ']'

        # if ephemeral constant
        else:
            standard = word[1:-count]

        # adjust count because we have assumed that
        # word[0], word[-1] = '[', ']'
        count -= 1

        # The remaining count is the number of primitives
        # that we have given all their children, so remove
        # them from the stack ...
        for _ in range(count):
            stack.pop()

        # and add on that many ending parethesis
        standard += ')' * count

        # if there is more to do, add a comma
        if len(stack) > 1:
            standard += ','

        return standard

    def _convert_lisp_to_standard_primitive(self, word: str,
                                            stack: List[str],
                                            convertion_dict: Dict[str,str] = primitive2function) -> str:
        """A portion of convert_lisp_to_standard, which deals
        with the terminals only.

        Parameters
        ----------
        word : str
            The terminal, possibly with extra bits. For example,
            word could be '[*'
        stack : list
            The stack of primitives encountered in the list so far.

        Returns
        -------
        standard : str
            The string word converted to standard, potentially
            containing some ) and/or a single ,
        """

        if word[1:] in convertion_dict:
            stack.append(convertion_dict[word[1:]])
            standard = stack[-1] + '('

        else:
            message = 'ERROR in Tree._convert_lisp_to_standard_primitive: bad function'+word[1:]
            raise UnknownPrimitiveError(message)

        return standard

    def convert_lisp_to_standard(self, convertion_dict: Dict[str,str] = None) -> str:
        """General versions of this function where
        conversion is specified by a dictionary.
        This function also forces input variables
        to be involved in the expression so that
        vectorization can be achieved (using eval).

        Parameters
        ----------
        convertion_dict : dict
            A dictionary explaining the convertion from
            keys to values. Some of the names in self.P
            (the primitive set) are short hand, so the
            convertion_dict can be used to make them long
            again. The key '#f' refers to ephemeral constants.

        Examples
        --------
        >>> tree = Tree('(* (3) (1))')
        >>> tree.get_lisp_string()
        (* (3) (1))

        >>> tree.convert_lisp_to_standard(convertion_dict={'*': 'Apple'})
        Apple(3,1)+0*x[0]
        """

        if convertion_dict is None:
            convertion_dict = primitive2function

        lisp = str(self.tree).replace(',', '').replace('\'', '')

        stack = ['']
        standard = ''

        split_lisp = lisp.split()

        # Check if single node function to avoid added comma at the end of expr
        if len(split_lisp) == 1:
            return self._convert_lisp_to_standard_terminal(lisp, stack)

        for word in split_lisp:

            if word[0] == '[' and word[-1] == ']':
                standard += self._convert_lisp_to_standard_terminal(word,
                                                                    stack)

            elif word[0] == '[':
                standard += self._convert_lisp_to_standard_primitive(word,
                                                                     stack,
                                                                     convertion_dict)

        if 'x' not in standard:  # assumes only x is for the variable
            var = 'x[0]'
            standard += '+0*' + var  # to avoid vectorization issue

        return standard

# ------------------------------------------------------------ #
#                      Build Trees
# ------------------------------------------------------------ #

    def from_string(self, expression: str,
                    actual_lisp: bool = False) -> None:
        """Construct a tree from a lisp string.
        The lisp should have parenthesis around
        terminal nodes to make this conversion
        easier. The tree is saved to self.tree
        as a list of lists.

        Parameters
        ----------
        expression : str
            The lisp string

        Examples
        --------
        The first two lines of this example can be performed
        with tree = Tree('(+ (x0) (1))')
        >>> tree = Tree(None)
        >>> tree.from_string('(+ (x0) (1))')
        >>> tree.get_lisp_string()
        (+ (x0) (1))

        >>> tree = Tree(None)
        >>> tree.from_string('(+ x0 (* c1 3))', actual_lisp=True)
        >>> tree.get_lisp_string()
        (+ (x0) (* (c1) (3)))

        >>> tree.get_lisp_string(actual_lisp=True)
        (+ x0 (* c1 3))
        """

        sbracket_expression = expression.replace('(', '[').replace(')', ']').replace(' ', ', ')

        for primitive in primitive2function.keys():

            if primitive != '-':
                sbracket_expression = sbracket_expression.replace(primitive+',', '\'' + primitive + '\',')

            else:
                # check if - is a negative sign rather than subtraction
                subtraction_index = [i for i, c in enumerate(sbracket_expression) if c == '-' and sbracket_expression[i+1] == ',']
                sbracket_expression = ''.join(['\'-\'' if i in subtraction_index else c for i, c in enumerate(sbracket_expression)])

        if actual_lisp:
            # find constants that give actual number
            string_list = []

            for word in sbracket_expression.split(' '):
                no_brackets = word.replace('[', '').replace(']', '')
                comma = False

                if ',' in word:
                    no_brackets = no_brackets.replace(',', '')
                    comma = True

                # is it a terminal
                try:
                    if len(no_brackets) == 1:
                        # is it a constant?
                        float(no_brackets)

                    else:
                        # is it a variable?
                        float(no_brackets[1:])

                # no, it is not a terminal (const or var)
                except ValueError:
                    string_list.append(word)

                # yes, it's a terminal
                else:
                    if comma:
                        to_append = '[' + word[:-1] + '],'
                    else:
                        to_append = '[' + word + ']'

                    string_list.append(to_append)

            sbracket_expression = ' '.join(string_list)

        for var_num in range(self.num_vars):
            sbracket_expression = sbracket_expression.replace('x'+str(var_num)+']', '\'x'+str(var_num)+'\']')

        self.tree = eval(sbracket_expression)

# ------------------------------------------------------------ #
#                   Get Tree Info
# ------------------------------------------------------------ #

    def select_subtree(self, child_indices: Tuple[int], subtree=None):
        """Find the node associated with the index list and return it.

        Parameters
        ----------
        child_indices : iterable
            The indices of children. For example, the child_indices
            of the 0-th child of the root node would be [0].
        subtree : list (of lists, default=self.tree)
            current subtree

        Examples
        --------
        >>> tree = Tree('(* (x0) (* (x1) (x2)))', num_vars=3)
        >>> tree.select_subtree(child_indices=(1, 0))
        ['x1']
        """

        subtree = self.tree if subtree is None else subtree

        for index in child_indices:
            # Add 1 because subtree[0] is the
            # label of that subtree's root.
            subtree = subtree[index+1]

        return subtree

    def set_subtree(self, new_subtree, child_indices: Tuple[int], subtree=None):
        """Find and set the node referenced by child_indices equal
        to new_node.

        Parameters
        ----------
        new_subtree : list (like self.tree)
            The subtree that will be placed in the desired location.
        child_indices : iterable (of ints)
            The location of the node that where new_subtree should
            be placed. Iterable is the indices of children.
            For example, the child_indices of the 0-th child of root
            node would be [0].  Root node is indicated by [] or ().
        subtree : list (default=self.tree)
            Current subtree the child_index_list is based on.

        Examples
        --------
        >>> tree = Tree('(* (x0) (* (x1) (x2)))', num_vars=3)
        >>> tree.set_subtree(new_subtree=['+', [1], [2]],
                             child_indices=(1, 0))
        >>> tree.get_lisp_string()
        (* (x0) (* (+ (1) (2)) (x2)))
        """

        subtree = self.tree if subtree is None else subtree

        for index in child_indices:
            # Add 1 because subtree[0] is the
            # label of that subtree's root.
            subtree = subtree[index+1]

        # [:] means the locations of stored values
        # of subtree will be changed. Thus, self.tree
        # will be effected by this assignment.
        subtree[:] = new_subtree

        return subtree

    def get_node_list(self, prefix: Tuple[int] = (),
                      node_list: List[Tuple[int]] = None,
                      subtree=None) -> List[Tuple[int]]:
        """Get a list of all nodes below subtree (including itself).
        Each node is represented by an iterable of child indices.
        For example, [0, 1, 1] refers to the 0-th child's 1-st child's
        1-st child. The parameters are mostly for recursion,
        and in most instances need not be specified.

        Parameters
        ----------
        prefix : tuple (default=())
            For recursion. This argument keeps track of the
            child indices. It is named prefix because it is
            appended to with the new child nodes.
        node_list : list (optional)
            For recursion. This argument keeps track of the
            nodes already found.
        subtree : list of lists (default=self.tree)
            The subtree to search through to find all nodes.

        Returns
        -------
        node_list : list (of tuples)
            The locations of all nodes in the tree as child
            indices. Note that the root node is empty tuple.

        Examples
        --------
        >>> tree = Tree('(- (3) (+ (x0) (3)))')
        >>> tree.get_node_list()
        [(), (0,), (1,), (1, 0), (1, 1)]
        """

        subtree = self.tree if subtree is None else subtree

        if node_list is None:
            node_list = [()]

        # if subtree is a single node
        if len(subtree) == 1:
            return node_list

        else:
            for i, st in enumerate(subtree):
                if type(st) == list:
                    node_list.append((*prefix, i-1))
                    node_list.extend(self.get_node_list(prefix=(*prefix, i-1), node_list=[], subtree=st))

            return node_list

    def get_num_leaves(self, num_leaves: int = 0,
                       num_nodes: int = 0, subtree=None) -> Tuple[int,int]:
        """Get a number of leaves in the tree and optionally the
        number nodes below subtree (including itself).
        Each node is represented by an iterable of child indices.
        For example, [0, 1, 1] refers to the 0-th child's 1-st child's
        1-st child. The parameters are mostly for recursion,
        and in most instances need not be specified.

        Parameters
        ----------
        num_nodes : list (optional)
            For recursion. This argument keeps track of the
            nodes already found.
        num_leaves : list (optional)
            For recursion. This argument keeps track of the
            leaves already found.
        subtree : list of lists (default=self.tree)
            The subtree to search through to find all nodes.

        Returns
        -------
        num_leaves : int
            The number of leaves in the tree
        num_nodes : int (optional)
            The number of nodes in the tree

        Examples
        --------
        >>> tree = Tree('(- (3) (+ (x0) (3)))')
        >>> num_leaves, num_nodes = tree.get_num_leaves()
        (3, 5)
        """

        subtree = self.tree if subtree is None else subtree

        # if subtree is a single node
        if len(subtree) == 1:
            return num_leaves+1, num_nodes+1

        else:
            for i, st in enumerate(subtree):
                if type(st) == list:
                    num_leaves, num_nodes = self.get_num_leaves(num_leaves=num_leaves, num_nodes=num_nodes, subtree=st)

            num_nodes += 1
            return num_leaves, num_nodes

    def get_node_dict(self, prefix: Tuple[int] = (),
                      node_dict: Dict[Tuple[int],str] = None, subtree=None) -> Dict[Tuple[int],str]:
        """Get a dictionary of all nodes in subtree. Each node is
        represented by a label and a list of child indices.
        The list [0, 1, 1] refers to the 0-th child's 1-st
        child's 1-st child. The parameters are mostly for
        recursion, and in most instances need not be
        specified.

        Parameters
        ----------
        prefix : tuple (default=())
            For recursion. This argument keeps track of the
            child indices. It is named prefix because it is
            appended to with the new child nodes.
        node_dict : dict (optional)
            For recursion. This argument keeps track of the
            nodes already found.
        subtree : list of lists (default=self.tree)
            The subtree to search through to find all nodes.

        Returns
        -------
        node_dict : list (of tuples)
            The locations of all nodesin the tree as child
            indices (these are the keys) and the node labels
            (these the values).

        Examples
        --------
        >>> tree = Tree('(- (3) (+ (x0) (3)))')
        >>> tree.get_node_dict()
        {(1, 0): 'x0', (): '-', (1,): '+', (0,): 3, (1, 1): 3}
        """

        subtree = self.tree if subtree is None else subtree

        if node_dict is None:
            node_dict = {(): subtree[0]}

        else:
            node_dict[prefix] = subtree[0]

        if len(subtree) == 1:
            return node_dict

        else:
            for i, st in enumerate(subtree):
                if type(st) == list:
                    self.get_node_dict(prefix=(*prefix, i - 1), subtree=st, node_dict=node_dict)

            return node_dict

    def get_node_map(self, loc: Tuple[int] = (),
                     node_dict: Dict[Union[str, float], Set[Tuple[int]]] = None,
                     subtree=None) -> Dict[Union[str, float], Set[Tuple[int]]]:
        """Look through all nodes and record nodes in a dictionary.
        This dictionary will fill all locations of node names in the tree.

        Parameters
        ----------
        loc : tuple (default=())
            For recursion. This argument keeps track of the
            locations.
        node_dict : dict (optional)
            For recursion. This argument keeps track of the
            nodes already found.
        subtree : list of lists (default=self.tree)
            The subtree to search through to find all nodes.

        Returns
        -------
        node_dict : list (of tuples)
            The node labels (these the keys). The locations
            of all nodes with the same label are stored in
            a set (these are the keys).

        Examples
        --------
        >>> tree = Tree('(- (3) (+ (x0) (3)))')
        >>> tree.get_node_map()
        {'-': {()}, 3: {(0,), (1, 1)}, 'x0': {(1, 0)}, '+': {(1,)}}
        """

        subtree = self.tree if subtree is None else subtree

        if node_dict is None:
            node_dict = {}

        if subtree[0] in node_dict:
            node_dict[subtree[0]] = node_dict[subtree[0]].union({loc})

        else:
            node_dict[subtree[0]] = {loc}  # set literal

        if len(subtree) == 1:
            return node_dict

        else:
            for i, st in enumerate(subtree):
                if type(st) == list:
                    self.get_node_map(loc=(*loc, i - 1), subtree=st, node_dict=node_dict)

            return node_dict

    def get_tree_size(self) -> int:
        """Count the number of nodes in the tree and return it.

        Returns
        -------
        tree_size : int
            The number of nodes in the tree.

        Examples
        --------
        >>> tree = Tree('(- (3) (+ (x0) (3)))')
        >>> tree.get_tree_size()
        5
        """
        node_labels = self.get_node_dict().values()
        return len(node_labels)

    def get_depth(self) -> int:
        """Find the deepest node in the tree and
        return the depth. This is done
        by looking at the list of all nodes.

        Returns
        -------
        max_depth : int
            The largest number of links between any
            node and the root node.

        Examples
        --------
        >>> tree = Tree('(- (3) (+ (x0) (3)))')
        >>> tree.get_depth()
        2
        """

        node_list = self.get_node_list()

        if node_list == [()]:
            return 0

        node_depth = [len(node_str) for node_str in node_list]
        return max(node_depth)

    def draw_tree(self, save_loc: str) -> None:
        """Draw tree using pygraphviz. Eventually,
        would like to make it possible to use LaTeX
        for the labels. There is a Python package
        (dot2tex).

        Parameters
        ----------
        save_loc : str
            The image generated will be saved
            in the location specified.
        """

        vis_tree = pgv.AGraph()

        node_dict = self.get_node_dict()

        # need to make sure that nodes are ordered for - or /
        m = max(map(len, node_dict.keys()))
        sort_key = lambda x: 2**(len(x[0])+m) + sum([2**xi for xi in x[0]])

        node_dict = collections.OrderedDict(sorted(node_dict.items(), key=sort_key))

        edgeColor = 'black'
        nodeColor = 'whitesmoke'

        for key in node_dict:
            if key == '':
                fake_key = 'root'
            else:
                fake_key = key

            try:
                float(node_dict[key])
            except ValueError:
                label = node_dict[key]
            else:
                label = '%.2E' % node_dict[key]

            vis_tree.add_node(fake_key, label=label,
                              fixedsize=False,
                              style='filled',
                              color=edgeColor,
                              shape='circle',
                              fillcolor=nodeColor)

            if key != '':
                if len(key) == 1:
                    parent = 'root'
                else:
                    parent = key[:-1]

                vis_tree.add_edge(parent, key)

        vis_tree.draw(save_loc, prog='dot')
