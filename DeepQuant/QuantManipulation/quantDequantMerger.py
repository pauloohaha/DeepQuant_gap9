# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Pu Deng <pudeng@iis.ee.ethz.ch>

import torch.fx as fx


BLUE = "\033[94m"
ENDC = "\033[0m"
CHECK = " ✓"
ARROW = " ›"

def quantDequantMerger(
    fxModel: fx.GraphModule, debug: bool = False
) -> fx.GraphModule:
    """
    This fucntion merge the unnecessary quant and dequant layers
    
    :param fxModel: input graph with redunant quant/dequant layers
    :type fxModel: fx.GraphModule
    :param debug:  If True, prints debug information.
    :type debug: bool

    :return: 
    :rtype: fx.GraphModule, dequant and quant layers that scale multiply to 1 will be removed
    """

    '''first merge quant dequant pair'''
    graph = fxModel.graph
    allNodes = list(graph.nodes)

    if debug:
        print(f"{BLUE}{ARROW} Starting Modification of Dequant Nodes...{ENDC}")

    for node in allNodes:
        if node.op != "call_module":
            continue
        nodeMod = fxModel.get_submodule(node.target)
        if nodeMod.__class__.__name__ == "Quant":
            user_node = list(node.users.keys())[0]
            if user_node.op != "call_module":
                continue
            
            userMod = fxModel.get_submodule(user_node.target)
            if(nodeMod.max_val != None):
                if(nodeMod.max_val + 1 != -1 * nodeMod.min_val):
                    # skip relu quant/dequant pairs
                    continue
            if userMod.__class__.__name__ == "Dequant":
                #next node is consecutive dequant
                if((userMod.scale == None and nodeMod.scale == None) or (userMod.scale == nodeMod.scale)):
                    #skip dummy nodes that don't have scale and nodes that have same scale
                    node_before_pair = node.args[0]
                    
                    user_node.replace_all_uses_with(node_before_pair)
                    
                    for usr in list(user_node.users.keys()):
                          user_node.users[usr] = None
                    for usr in list(node.users.keys()):
                          node.users[usr] = None

                    pass
  
    graph.lint()
    graph.eliminate_dead_code()

    # Remove submodules that are now unused
    fxModel.delete_all_unused_submodules()

    # Recompile so that the generated forward code no longer references removed nodes
    fxModel.recompile()


    '''then merge dequant quant pair'''
    graph = fxModel.graph
    allNodes = list(graph.nodes)

    for node in allNodes:
        if node.op != "call_module":
            continue
        nodeMod = fxModel.get_submodule(node.target)
        if nodeMod.__class__.__name__ == "Dequant":
            user_node = list(node.users.keys())[0]
            if user_node.op != "call_module":
                continue
            userMod = fxModel.get_submodule(user_node.target)

            if userMod.__class__.__name__ == "Quant":
                if(userMod.max_val != None):
                    if(userMod.max_val + 1 != -1 * userMod.min_val):
                        # skip relu quant/dequant pairs
                        continue
            
                #next node is consecutive dequant
                if((userMod.scale == None and nodeMod.scale == None) or (userMod.scale == nodeMod.scale)):
                    #skip dummy nodes that don't have scale and nodes that have same scale
                    node_before_pair = node.args[0]
                    
                    user_node.replace_all_uses_with(node_before_pair)
                    
                    for usr in list(user_node.users.keys()):
                          user_node.users[usr] = None
                    for usr in list(node.users.keys()):
                          node.users[usr] = None

                    pass
  
    graph.lint()
    graph.eliminate_dead_code()

    # Remove submodules that are now unused
    fxModel.delete_all_unused_submodules()

    # Recompile so that the generated forward code no longer references removed nodes
    fxModel.recompile()

    return fxModel