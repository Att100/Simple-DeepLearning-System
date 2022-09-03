from typing import Any

from TinyDL.core.autograd.variable import Variable


class DyGraphTracer(object):
    """
    # DyGraphTracer

    ## Attributes
            structure (dict): store the structure of a dynamic computation graph
            name (str): the name of this record
            n_node (int): the number of nodes in the computation graph
            mode (str): layer mode of graph node mode

    ## Args
            name (str): the name of this record
            mode (str): layer mode of graph node mode
    """
    def __init__(self, name: str = "", mode='graph') -> None:
        super().__init__()
        self.structure: dict = {'name': name, 'input': []}
        self.name: str = name
        self.n_node: int = 1
        self.mode: str = mode

    def step(self, node = None):
        """
        record a node 
        """
        if node is None:
            return
        if self.mode == 'graph':
            self._step_variable(node)
        elif self.mode == 'layer':
            self._step_layer(node)
            
    def _step_variable(self, node: Variable = None):
        node_info = {}
        if not node.requires_grad:
            return
        if node.is_leaf:
            self.structure['input'].append(node.name)
            return 
        node.name = "node_{}".format(self.n_node)
        node_info['name'] = node.name
        if node.operator is not None:
            node_info['type'] = node.operator.name
        node_info['bottom'] = []
        if node.l_child is not None:
            node_info['bottom'].append(node.l_child.name)
        if node.r_child is not None:
            node_info['bottom'].append(node.r_child.name)
        node_info['top'] = node.name
        if node.operator_param != None:
            node_info['{}_param'.format(node_info['type'].lower())] = node.operator_param
        self.structure[node.name] = node_info
        self.n_node += 1

    def _step_layer(self, node: Any = None):
        pass

    def to_proto(self) -> list:
        """
        transform the structure to prototxt string list
        """
        string_list = []
        space = "  "
        for k, v in self.structure.items():
            if k == 'name':
                if v != "":
                    string_list.append('name: "{}"'.format(v))
                    string_list.append("")
            elif k == 'input':
                for item in v:
                    string_list.append('input: "{}"'.format(item))
                string_list.append("")
            else:
                string_list.append("layer {")
                for k_, v_ in self.structure[k].items():
                    if isinstance(v_, str):
                        string_list.append(space+'{}: "{}"'.format(k_, v_))
                    elif isinstance(v_, list):
                        for item in v_:
                            string_list.append(space+'{}: "{}"'.format(k_, item))
                    elif isinstance(v_, dict):
                        string_list.append(space + k_+" {")
                        for k_p, v_p in v_.items():
                            string_list.append(space*2 + "{}: {}".format(k_p, v_p))
                        string_list.append(space + "}")
                string_list.append("}")
        return string_list
                
    def save_prototxt(self, path: str = ""):
        """
        save prototxt string list to .prototxt

        args:
            path (str): the path with the name of the prototxt file
        """
        string_list = self.to_proto()
        f = open(path, mode='w')
        for line in string_list:
            f.write(line + '\n')
        f.close()