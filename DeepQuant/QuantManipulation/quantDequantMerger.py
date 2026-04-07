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


def mergeReLURequant(
    fxModel: fx.GraphModule, debug: bool = False
) -> fx.GraphModule:
    """
    Merge dequant(0)->quant(1)->dequant(2)->quant(3) chains into
    dequant(0)->merged_quant(1) where all nodes have zero_point==0,
    quant(1) and quant(3) are unsigned with matching min/max range.

    The merged quant scale becomes: s_q1 * s_q3 / s_d2
    """
    graph = fxModel.graph
    allNodes = list(graph.nodes)

    if debug:
        print(f"{BLUE}{ARROW} Starting ReLU Requant Merging...{ENDC}")

    merge_count = 0

    for node in allNodes:
        if node.op != "call_module":
            continue

        # --- Candidate for dequant(0) ---
        nodeMod = fxModel.get_submodule(node.target)
        if nodeMod.__class__.__name__ != "Dequant":
            continue
        if nodeMod.zero_point != 0:
            continue

        # --- For each ReLU user of dequant(0), try to merge the chain ---
        for relu_node in list(node.users.keys()):
            if relu_node.op != "call_module":
                continue
            reluMod = fxModel.get_submodule(relu_node.target)
            if reluMod.__class__.__name__ != "InnerForwardImplWrapperActivation":
                continue
            if len(relu_node.users) != 1:
                continue

            # --- Follow to quant(1) (ReLU output quant, unsigned) ---
            q1_node = list(relu_node.users.keys())[0]
            if q1_node.op != "call_module":
                continue
            q1Mod = fxModel.get_submodule(q1_node.target)
            if q1Mod.__class__.__name__ != "Quant":
                continue
            if q1Mod.zero_point != 0:
                continue
            if q1Mod.min_val != 0:  # must be unsigned (ReLU)
                continue
            if len(q1_node.users) != 1:
                continue

            # --- Follow to dequant(2) ---
            d2_node = list(q1_node.users.keys())[0]
            if d2_node.op != "call_module":
                continue
            d2Mod = fxModel.get_submodule(d2_node.target)
            if d2Mod.__class__.__name__ != "Dequant":
                continue
            if d2Mod.zero_point != 0:
                continue
            if len(d2_node.users) != 1:
                continue

            # --- Follow to quant(3) ---
            q3_node = list(d2_node.users.keys())[0]
            if q3_node.op != "call_module":
                continue
            q3Mod = fxModel.get_submodule(q3_node.target)
            if q3Mod.__class__.__name__ != "Quant":
                continue
            if q3Mod.zero_point != 0:
                continue
            if q3Mod.min_val != 0:  # must be unsigned
                continue

            # --- Verify matching unsigned ranges ---
            if q1Mod.min_val != q3Mod.min_val or q1Mod.max_val != q3Mod.max_val:
                continue

            # --- All conditions met: merge ---
            # Merge quant(1), dequant(2), quant(3) into quant(1)
            # New scale: s_q1 * s_q3 / s_d2
            s_q1 = q1Mod.scale
            s_d2 = d2Mod.scale
            s_q3 = q3Mod.scale
            s_new = s_q1 * s_q3 / s_d2

            # Update quant(1) with merged scale; min_val/max_val/bit_width stay
            q1Mod.scale = s_new

            # Reroute quant(3)'s users to quant(1), then remove dequant(2) and quant(3)
            q3_node.replace_all_uses_with(q1_node)

            for usr in list(q3_node.users.keys()):
                q3_node.users[usr] = None
            for usr in list(d2_node.users.keys()):
                d2_node.users[usr] = None

            merge_count += 1

            if debug:
                print(f"{BLUE}{CHECK} Merged: {node.target} -> {q1_node.target} "
                      f"(removed {d2_node.target}, {q3_node.target})")
                print(f"       s_q1={s_q1}, s_d2={s_d2}, s_q3={s_q3} -> s_new={s_new}{ENDC}")

    graph.lint()
    graph.eliminate_dead_code()
    fxModel.delete_all_unused_submodules()
    fxModel.recompile()

    if debug:
        print(f"{BLUE}{CHECK} ReLU Requant Merging: {merge_count} chains merged{ENDC}")

    return fxModel


def mergeActivationRequant(
    fxModel: fx.GraphModule, debug: bool = False
) -> fx.GraphModule:
    """
    Merge InnerForwardImplWrapperActivation -> Quant -> Dequant -> Quant chains
    into InnerForwardImplWrapperActivation -> Quant (with merged scale).

    Both Quant nodes must be unsigned int8 with min_val=0, max_val=255.
    The merged quant scale becomes: s_q2 * s_q4 / s_d3
    """
    graph = fxModel.graph
    allNodes = list(graph.nodes)

    if debug:
        print(f"{BLUE}{ARROW} Starting Activation Requant Merging...{ENDC}")

    merge_count = 0

    for node in allNodes:
        if node.op != "call_module":
            continue

        # --- Node 1: InnerForwardImplWrapperActivation ---
        nodeMod = fxModel.get_submodule(node.target)
        if nodeMod.__class__.__name__ != "InnerForwardImplWrapperActivation":
            continue

        for q2_node in list(node.users.keys()):
            if q2_node.op != "call_module":
                continue

            # --- Node 2: Quant (unsigned, min=0, max=255) ---
            q2Mod = fxModel.get_submodule(q2_node.target)
            if q2Mod.__class__.__name__ != "Quant":
                continue
            if q2Mod.zero_point != 0 or q2Mod.min_val != 0 or q2Mod.max_val != 255:
                continue
            if len(q2_node.users) != 1:
                continue

            # --- Node 3: Dequant ---
            d3_node = list(q2_node.users.keys())[0]
            if d3_node.op != "call_module":
                continue
            d3Mod = fxModel.get_submodule(d3_node.target)
            if d3Mod.__class__.__name__ != "Dequant":
                continue
            if d3Mod.zero_point != 0:
                continue
            if len(d3_node.users) != 1:
                continue

            # --- Node 4: Quant (unsigned, min=0, max=255) ---
            q4_node = list(d3_node.users.keys())[0]
            if q4_node.op != "call_module":
                continue
            q4Mod = fxModel.get_submodule(q4_node.target)
            if q4Mod.__class__.__name__ != "Quant":
                continue
            if q4Mod.zero_point != 0 or q4Mod.min_val != 0 or q4Mod.max_val != 255:
                continue

            # --- Verify matching ranges ---
            if q2Mod.min_val != q4Mod.min_val or q2Mod.max_val != q4Mod.max_val:
                continue

            # --- All conditions met: merge ---
            s_q2 = q2Mod.scale
            s_d3 = d3Mod.scale
            s_q4 = q4Mod.scale
            s_new = s_q2 * s_q4 / s_d3

            q2Mod.scale = s_new

            # Reroute q4's users to q2, remove d3 and q4
            q4_node.replace_all_uses_with(q2_node)

            for usr in list(q4_node.users.keys()):
                q4_node.users[usr] = None
            for usr in list(d3_node.users.keys()):
                d3_node.users[usr] = None

            merge_count += 1

            if debug:
                print(f"{BLUE}{CHECK} Merged: {node.target} -> {q2_node.target} "
                      f"(removed {d3_node.target}, {q4_node.target})")
                print(f"       s_q2={s_q2}, s_d3={s_d3}, s_q4={s_q4} -> s_new={s_new}{ENDC}")

    graph.lint()
    graph.eliminate_dead_code()
    fxModel.delete_all_unused_submodules()
    fxModel.recompile()

    if debug:
        print(f"{BLUE}{CHECK} Activation Requant Merging: {merge_count} chains merged{ENDC}")

    return fxModel