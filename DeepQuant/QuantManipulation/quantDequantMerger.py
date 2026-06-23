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

def quantVoidDequantMerger(
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
                if((userMod.scale == None and nodeMod.scale == None)):
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


def quantDequantChainMerger(
    fxModel: fx.GraphModule, debug: bool = False
) -> fx.GraphModule:
    """
    Merge same-scale dequant(A)->quant(B) pairs only when followed by another
    dequant(C)->quant(D) pair, ensuring a quant/dequant boundary is always
    preserved between gemm output and the next kernel.
    """
    graph = fxModel.graph
    allNodes = list(graph.nodes)

    if debug:
        print(f"{BLUE}{ARROW} Starting Chain Dequant-Quant Merging...{ENDC}")

    merge_count = 0

    for nodeA in allNodes:
        if nodeA.op != "call_module":
            continue
        modA = fxModel.get_submodule(nodeA.target)
        if modA.__class__.__name__ != "Dequant":
            continue
        if len(nodeA.users) != 1:
            continue

        # --- quant(B): same scale as dequant(A), signed (not ReLU) ---
        nodeB = list(nodeA.users.keys())[0]
        if nodeB.op != "call_module":
            continue
        modB = fxModel.get_submodule(nodeB.target)
        if modB.__class__.__name__ != "Quant":
            continue
        if modB.max_val != None:
            if modB.max_val + 1 != -1 * modB.min_val:
                # skip relu quant/dequant pairs
                continue
        if not ((modA.scale == None and modB.scale == None) or (modA.scale == modB.scale)):
            continue
        if len(nodeB.users) != 1:
            continue

        # --- dequant(C) follows quant(B) ---
        nodeC = list(nodeB.users.keys())[0]
        if nodeC.op != "call_module":
            continue
        modC = fxModel.get_submodule(nodeC.target)
        if modC.__class__.__name__ != "Dequant":
            continue

        # --- quant(D) follows dequant(C) — just confirm it exists ---
        has_quant_user = False
        for userD in nodeC.users:
            if userD.op == "call_module":
                modD = fxModel.get_submodule(userD.target)
                if modD.__class__.__name__ == "Quant":
                    has_quant_user = True
                    break
        if not has_quant_user:
            continue

        # --- All conditions met: remove dequant(A) and quant(B) ---
        node_before_A = nodeA.args[0]
        nodeB.replace_all_uses_with(node_before_A)

        for usr in list(nodeB.users.keys()):
            nodeB.users[usr] = None
        for usr in list(nodeA.users.keys()):
            nodeA.users[usr] = None

        merge_count += 1

        if debug:
            print(f"{BLUE}{CHECK} Merged chain: removed {nodeA.target}, {nodeB.target} "
                  f"(kept {nodeC.target} -> {userD.target}){ENDC}")

    graph.lint()
    graph.eliminate_dead_code()
    fxModel.delete_all_unused_submodules()
    fxModel.recompile()

    if debug:
        print(f"{BLUE}{CHECK} Chain Dequant-Quant Merging: {merge_count} pairs merged{ENDC}")

    return fxModel


def quantDequantChainMergerSecond(
    fxModel: fx.GraphModule, debug: bool = False
) -> fx.GraphModule:
    """
    Detect dequant(A)->quant(B)->dequant(C)->quant(D) chains and merge the
    second pair dequant(C)->quant(D) if they have the same scale.
    """
    graph = fxModel.graph
    allNodes = list(graph.nodes)

    if debug:
        print(f"{BLUE}{ARROW} Starting Chain Dequant-Quant Merging (second pair)...{ENDC}")

    merge_count = 0

    for nodeA in allNodes:
        if nodeA.op != "call_module":
            continue
        modA = fxModel.get_submodule(nodeA.target)
        if modA.__class__.__name__ != "Dequant":
            continue
        if len(nodeA.users) != 1:
            continue

        # --- quant(B): confirm first pair exists, skip ReLU ---
        nodeB = list(nodeA.users.keys())[0]
        if nodeB.op != "call_module":
            continue
        modB = fxModel.get_submodule(nodeB.target)
        if modB.__class__.__name__ != "Quant":
            continue
        if modB.max_val != None:
            if modB.max_val + 1 != -1 * modB.min_val:
                continue
        if len(nodeB.users) != 1:
            continue

        # --- dequant(C) ---
        nodeC = list(nodeB.users.keys())[0]
        if nodeC.op != "call_module":
            continue
        modC = fxModel.get_submodule(nodeC.target)
        if modC.__class__.__name__ != "Dequant":
            continue
        if len(nodeC.users) != 1:
            continue

        # --- quant(D): same scale as dequant(C), skip ReLU ---
        nodeD = list(nodeC.users.keys())[0]
        if nodeD.op != "call_module":
            continue
        modD = fxModel.get_submodule(nodeD.target)
        if modD.__class__.__name__ != "Quant":
            continue
        if modD.max_val != None:
            if modD.max_val + 1 != -1 * modD.min_val:
                continue
        if not ((modC.scale == None and modD.scale == None) or (modC.scale == modD.scale)):
            continue

        # --- All conditions met: remove dequant(C) and quant(D) ---
        nodeD.replace_all_uses_with(nodeB)

        for usr in list(nodeD.users.keys()):
            nodeD.users[usr] = None
        for usr in list(nodeC.users.keys()):
            nodeC.users[usr] = None

        merge_count += 1

        if debug:
            print(f"{BLUE}{CHECK} Merged chain (2nd): removed {nodeC.target}, {nodeD.target} "
                  f"(kept {nodeA.target} -> {nodeB.target}){ENDC}")

    graph.lint()
    graph.eliminate_dead_code()
    fxModel.delete_all_unused_submodules()
    fxModel.recompile()

    if debug:
        print(f"{BLUE}{CHECK} Chain Dequant-Quant Merging (second pair): {merge_count} pairs merged{ENDC}")

    return fxModel


def quantDequantReLUChainMerger(
    fxModel: fx.GraphModule, debug: bool = False
) -> fx.GraphModule:
    """
    Detect dequant(1)->quant(2)->dequant(3)->ReLU(4)->quant(5) chains and merge
    quant(2) and dequant(3) if they have the same scale. This connects
    dequant(1) directly to ReLU(4).
    """
    graph = fxModel.graph
    allNodes = list(graph.nodes)

    if debug:
        print(f"{BLUE}{ARROW} Starting Chain Dequant-Quant-ReLU Merging...{ENDC}")

    merge_count = 0

    for node1 in allNodes:
        if node1.op != "call_module":
            continue
        mod1 = fxModel.get_submodule(node1.target)
        if mod1.__class__.__name__ != "Dequant":
            continue
        if len(node1.users) != 1:
            continue

        # --- quant(2): not ReLU ---
        node2 = list(node1.users.keys())[0]
        if node2.op != "call_module":
            continue
        mod2 = fxModel.get_submodule(node2.target)
        if mod2.__class__.__name__ != "Quant":
            continue
        if mod2.max_val != None:
            if mod2.max_val + 1 != -1 * mod2.min_val:
                continue
        if len(node2.users) != 1:
            continue

        # --- dequant(3): same scale as quant(2) ---
        node3 = list(node2.users.keys())[0]
        if node3.op != "call_module":
            continue
        mod3 = fxModel.get_submodule(node3.target)
        if mod3.__class__.__name__ != "Dequant":
            continue
        if not ((mod2.scale == None and mod3.scale == None) or (mod2.scale == mod3.scale)):
            continue
        if len(node3.users) != 1:
            continue

        # --- ReLU(4) (InnerForwardImplWrapperActivation) ---
        node4 = list(node3.users.keys())[0]
        if node4.op != "call_module":
            continue
        mod4 = fxModel.get_submodule(node4.target)
        if mod4.__class__.__name__ != "InnerForwardImplWrapperActivation":
            continue
        if len(node4.users) != 1:
            continue

        # --- quant(5) follows ReLU — just confirm it exists ---
        node5 = list(node4.users.keys())[0]
        if node5.op != "call_module":
            continue
        mod5 = fxModel.get_submodule(node5.target)
        if mod5.__class__.__name__ != "Quant":
            continue

        # --- All conditions met: remove quant(2) and dequant(3) ---
        # Connect dequant(1) directly to ReLU(4)
        node3.replace_all_uses_with(node1)

        for usr in list(node3.users.keys()):
            node3.users[usr] = None
        for usr in list(node2.users.keys()):
            node2.users[usr] = None

        merge_count += 1

        if debug:
            print(f"{BLUE}{CHECK} Merged ReLU chain: removed {node2.target}, {node3.target} "
                  f"(kept {node1.target} -> ReLU -> {node5.target}){ENDC}")

    graph.lint()
    graph.eliminate_dead_code()
    fxModel.delete_all_unused_submodules()
    fxModel.recompile()

    if debug:
        print(f"{BLUE}{CHECK} Chain Dequant-Quant-ReLU Merging: {merge_count} pairs merged{ENDC}")

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


def mergeInputQuantDequant(
    fxModel: fx.GraphModule, debug: bool = False
) -> fx.GraphModule:
    """
    Remove redundant `placeholder -> Quant -> Dequant` pairs at the graph input.

    The placeholder carries fp32 data, so if the quant and the immediately
    following dequant share the same scale and zero_point, the round-trip
    fp32 -> int -> fp32 is an identity and both nodes can be dropped,
    feeding the placeholder directly into the original users of the dequant.
    """
    graph = fxModel.graph
    allNodes = list(graph.nodes)

    if debug:
        print(f"{BLUE}{ARROW} Starting Input Quant-Dequant Merging...{ENDC}")

    merge_count = 0

    for ph_node in allNodes:
        if ph_node.op != "placeholder":
            continue

        for q_node in list(ph_node.users.keys()):
            if q_node.op != "call_module":
                continue
            qMod = fxModel.get_submodule(q_node.target)
            if qMod.__class__.__name__ != "Quant":
                continue
            if len(q_node.users) != 1:
                continue

            d_node = list(q_node.users.keys())[0]
            if d_node.op != "call_module":
                continue
            dMod = fxModel.get_submodule(d_node.target)
            if dMod.__class__.__name__ != "Dequant":
                continue

            scales_match = (
                (qMod.scale is None and dMod.scale is None)
                or (qMod.scale == dMod.scale)
            )
            if not scales_match:
                continue
            if qMod.zero_point != dMod.zero_point:
                continue

            d_node.replace_all_uses_with(ph_node)

            for usr in list(d_node.users.keys()):
                d_node.users[usr] = None
            for usr in list(q_node.users.keys()):
                q_node.users[usr] = None

            merge_count += 1

            if debug:
                print(f"{BLUE}{CHECK} Merged input: {ph_node.name} -> "
                      f"removed {q_node.target}, {d_node.target} "
                      f"(scale={qMod.scale}, zp={qMod.zero_point}){ENDC}")

    graph.lint()
    graph.eliminate_dead_code()
    fxModel.delete_all_unused_submodules()
    fxModel.recompile()

    if debug:
        print(f"{BLUE}{CHECK} Input Quant-Dequant Merging: {merge_count} pairs merged{ENDC}")

    return fxModel


def quantDequantChainMergerMiddle(
    fxModel: fx.GraphModule, debug: bool = False
) -> fx.GraphModule:
    """
    Detect `Dequant(1) -> Quant(2) -> Dequant(3) -> Quant(4)` chains and drop
    the middle `Quant(2) -> Dequant(3)` pair when they share the same scale.

    Since Quant(2) and Dequant(3) have the same scale, the round-trip
    fp32 -> int -> fp32 is an identity, so the chain collapses to
    `Dequant(1) -> Quant(4)`.
    """
    graph = fxModel.graph

    if debug:
        print(f"{BLUE}{ARROW} Starting Chain Dequant-Quant Middle Merging...{ENDC}")

    merge_count = 0
    changed = True

    # Restart the scan after every merge so that dequant nodes that shifted
    # position in the chain (because an earlier pair was removed) get another
    # chance to match. Terminates when a full pass finds nothing to merge.
    while changed:
        changed = False
        for node1 in list(graph.nodes):
            if node1.op != "call_module":
                continue
            mod1 = fxModel.get_submodule(node1.target)
            if mod1.__class__.__name__ != "Dequant":
                continue
            if len(node1.users) != 1:
                continue

            # --- quant(2) ---
            node2 = list(node1.users.keys())[0]
            if node2.op != "call_module":
                continue
            mod2 = fxModel.get_submodule(node2.target)
            if mod2.__class__.__name__ != "Quant":
                continue
            if len(node2.users) != 1:
                continue

            # --- dequant(3): same scale as quant(2) ---
            node3 = list(node2.users.keys())[0]
            if node3.op != "call_module":
                continue
            mod3 = fxModel.get_submodule(node3.target)
            if mod3.__class__.__name__ != "Dequant":
                continue
            if not ((mod2.scale == None and mod3.scale == None) or (mod2.scale == mod3.scale)):
                continue
            if len(node3.users) != 1:
                continue

            # --- quant(4) follows dequant(3) ---
            node4 = list(node3.users.keys())[0]
            if node4.op != "call_module":
                continue
            mod4 = fxModel.get_submodule(node4.target)
            if mod4.__class__.__name__ != "Quant":
                continue

            # --- All conditions met: remove quant(2) and dequant(3) ---
            # Connect dequant(1) directly to quant(4)
            node3.replace_all_uses_with(node1)

            for usr in list(node3.users.keys()):
                node3.users[usr] = None
            for usr in list(node2.users.keys()):
                node2.users[usr] = None

            merge_count += 1
            changed = True

            if debug:
                print(f"{BLUE}{CHECK} Merged middle: removed {node2.target}, {node3.target} "
                      f"(kept {node1.target} -> {node4.target}){ENDC}")

            # Restart the outer scan so earlier Dequants whose downstream
            # chain has just shifted get another chance to match.
            break

    graph.lint()
    graph.eliminate_dead_code()
    fxModel.delete_all_unused_submodules()
    fxModel.recompile()

    if debug:
        print(f"{BLUE}{CHECK} Chain Dequant-Quant Middle Merging: {merge_count} pairs merged{ENDC}")

    return fxModel


def mergeTCneighborgatherDequantQuant(
    fxModel: fx.GraphModule, debug: bool = False
) -> fx.GraphModule:
    """
    Detect `tc_neighbor_gather -> Dequant -> Quant` and drop the Dequant/Quant
    pair when they share scale, zero_point, and bit_width. Because the gather
    is already in the integer domain, the round-trip int->fp32->int collapses
    to identity and tc_neighbor_gather can feed the Quant's users directly.
    """
    graph = fxModel.graph

    if debug:
        print(f"{BLUE}{ARROW} Starting TCneighborgather Dequant-Quant Merging...{ENDC}")

    merge_count = 0

    for node_g in list(graph.nodes):
        if node_g.op != "call_module" or ".tc_neighbor_gather" not in node_g.target:
            continue
        if len(node_g.users) != 1:
            continue

        node_d = next(iter(node_g.users))
        if node_d.op != "call_module":
            continue
        mod_d = fxModel.get_submodule(node_d.target)
        if mod_d.__class__.__name__ != "Dequant":
            continue
        if len(node_d.users) != 1:
            continue

        node_q = next(iter(node_d.users))
        if node_q.op != "call_module":
            continue
        mod_q = fxModel.get_submodule(node_q.target)
        if mod_q.__class__.__name__ != "Quant":
            continue

        if mod_d.scale is None and mod_q.scale is None:
            same_scale = True
        elif mod_d.scale is None or mod_q.scale is None:
            same_scale = False
        else:
            # Accept scales within 1% relative tolerance (|d/q - 1| < 0.01).
            same_scale = abs(float(mod_d.scale) / float(mod_q.scale) - 1.0) < 0.01
        same_zp = mod_d.zero_point == mod_q.zero_point
        same_bit_width = mod_d.bit_width == mod_q.bit_width
        if not (same_scale and same_zp and same_bit_width):
            continue

        node_q.replace_all_uses_with(node_g)

        for usr in list(node_q.users.keys()):
            node_q.users[usr] = None
        for usr in list(node_d.users.keys()):
            node_d.users[usr] = None

        merge_count += 1

        if debug:
            print(f"{BLUE}{CHECK} Merged tc_neighbor_gather -> dequant -> quant: "
                  f"removed {node_d.target}, {node_q.target} (kept {node_g.target}){ENDC}")

    graph.lint()
    graph.eliminate_dead_code()
    fxModel.delete_all_unused_submodules()
    fxModel.recompile()

    if debug:
        print(f"{BLUE}{CHECK} TCneighborgather Dequant-Quant Merging: {merge_count} pairs merged{ENDC}")

    return fxModel


def mergeAddDequantQuantIntoDequant(
    fxModel: fx.GraphModule, debug: bool = False
) -> fx.GraphModule:
    """
    Detect `Add(1) -> Dequant(2) -> Quant(3) -> Dequant(4)` and fold the
    middle Dequant(2)/Quant(3) into Dequant(4).

    Requires Add(1), Dequant(2), Quant(3) to each have exactly one user.
    Updates Dequant(4) in place with new scale `s_d2 * s_d4 / s_q3`
    (zero_points must be 0, matching the convention used by the other
    requant mergers in this file).
    """
    graph = fxModel.graph

    if debug:
        print(f"{BLUE}{ARROW} Starting Add Dequant-Quant -> Dequant Merging...{ENDC}")

    merge_count = 0

    for node_add in list(graph.nodes):
        if node_add.op != "call_function" or "add" not in node_add.name:
            continue
        if len(node_add.users) != 1:
            continue

        node_d2 = next(iter(node_add.users))
        if node_d2.op != "call_module":
            continue
        mod_d2 = fxModel.get_submodule(node_d2.target)
        if mod_d2.__class__.__name__ != "Dequant":
            continue
        if len(node_d2.users) != 1:
            continue

        node_q3 = next(iter(node_d2.users))
        if node_q3.op != "call_module":
            continue
        mod_q3 = fxModel.get_submodule(node_q3.target)
        if mod_q3.__class__.__name__ != "Quant":
            continue
        if len(node_q3.users) != 1:
            continue

        node_d4 = next(iter(node_q3.users))
        if node_d4.op != "call_module":
            continue
        mod_d4 = fxModel.get_submodule(node_d4.target)
        if mod_d4.__class__.__name__ != "Dequant":
            continue

        if mod_d2.scale is None or mod_q3.scale is None or mod_d4.scale is None:
            continue
        if mod_d2.zero_point != 0 or mod_q3.zero_point != 0 or mod_d4.zero_point != 0:
            continue

        mod_d4.scale = float(mod_d2.scale) * float(mod_d4.scale) / float(mod_q3.scale)

        new_args = list(node_d4.args)
        for i, a in enumerate(new_args):
            if a is node_q3:
                new_args[i] = node_add
        node_d4.args = tuple(new_args)

        for usr in list(node_q3.users.keys()):
            node_q3.users[usr] = None
        for usr in list(node_d2.users.keys()):
            node_d2.users[usr] = None

        merge_count += 1

        if debug:
            print(f"{BLUE}{CHECK} Merged add -> dequant -> quant -> dequant: "
                  f"folded {node_d2.target}, {node_q3.target} into {node_d4.target} "
                  f"(new scale={mod_d4.scale}){ENDC}")

    graph.lint()
    graph.eliminate_dead_code()
    fxModel.delete_all_unused_submodules()
    fxModel.recompile()

    if debug:
        print(f"{BLUE}{CHECK} Add Dequant-Quant -> Dequant Merging: {merge_count} pairs merged{ENDC}")

    return fxModel
