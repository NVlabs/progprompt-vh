# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


def get_obj_ids_for_adding_states(graph):
    wallphone_ids = [node['id'] for node in graph['nodes'] if node["class_name"] == "wallphone"]
    microwave_ids = [node['id'] for node in graph['nodes'] if node["class_name"] == "microwave"]
    stove_ids = [node['id'] for node in graph['nodes'] if node["class_name"] == "stove"]
    fryingpan_ids = [node['id'] for node in graph['nodes'] if node["class_name"] == "fryingpan"]
    washingmachine_ids = [node['id'] for node in graph['nodes'] if node["class_name"] == "washingmachine"]
    sink_ids = [node['id'] for node in graph['nodes'] if node["class_name"] == "sink"]
    faucet_ids = [node['id'] for node in graph['nodes'] if node["class_name"] == "faucet"]
    ## toaster condition - TODO ##
    return (wallphone_ids, microwave_ids, stove_ids, fryingpan_ids, washingmachine_ids, sink_ids, faucet_ids)


def add_additional_obj_states(state, ids, nodes_with_additional_states):
    ## TODO fix faucet, stove, washingmachine
    wallphone_ids, microwave_ids, stove_ids, fryingpan_ids, washingmachine_ids, sink_ids, faucet_ids = ids
    wallphone_cond = [idx for idx in range(len(state['nodes'])) if state['nodes'][idx]['id'] in wallphone_ids and 'ON' in  state['nodes'][idx]['states']]
    for i in wallphone_cond:
        state['nodes'][i]['states'].append("USED")
        nodes_with_additional_states[state['nodes'][i]['id']] = state['nodes'][i]
    
    microwave_cond = [state['nodes'][idx]['id'] for idx in range(len(state['nodes'])) if state['nodes'][idx]['id'] in microwave_ids and 'ON' in state['nodes'][idx]['states']]
    if len(microwave_cond)>0:
        food_in_microwave = [n['from_id'] for n in state["edges"] if n['to_id'] in microwave_cond and n["relation_type"] == "INSIDE"]
        food_in_microwave_cond = [idx for idx in range(len(state['nodes'])) if state['nodes'][idx]['id'] in food_in_microwave and state['nodes'][idx]['category'] == 'Food']
        for i in food_in_microwave_cond:
            state['nodes'][i]['states'].append("HEATED")
            nodes_with_additional_states[state['nodes'][i]['id']] = state['nodes'][i]

    stove_cond = [state['nodes'][idx]['id'] for idx in range(len(state['nodes'])) if state['nodes'][idx]['id'] in stove_ids and 'ON' in state['nodes'][idx]['states']]
    if len(stove_cond)>0:
        # print("stove on")
        fryingpan_on_stove = [n['from_id'] for n in state["edges"] if n['to_id'] in stove_cond and n['from_id'] in fryingpan_ids and n["relation_type"] == "ON"]
        if len(fryingpan_on_stove)>0:
            # print("pan on stove")
            food_in_fryingpan = [n['from_id'] for n in state["edges"] if n['to_id'] in fryingpan_on_stove and n["relation_type"] == "ON"]
            food_in_fryingpan_cond = [idx for idx in range(len(state['nodes'])) if state['nodes'][idx]['id'] in food_in_fryingpan and state['nodes'][idx]['category'] == 'Food']
            for i in food_in_fryingpan_cond:
                # print("food in pan")
                state['nodes'][i]['states'].append("HEATED")
                nodes_with_additional_states[state['nodes'][i]['id']] = state['nodes'][i]

    washingmachine_cond = [state['nodes'][idx]['id'] for idx in range(len(state['nodes'])) if state['nodes'][idx]['id'] in washingmachine_ids and 'ON' in state['nodes'][idx]['states']]
    if len(washingmachine_cond)>0:
        cloth_in_washingmachine = [n['from_id'] for n in state["edges"] if n['to_id'] in washingmachine_cond and n["relation_type"] == "INSIDE"]
        cloth_in_washingmachine_cond = [idx for idx in range(len(state['nodes'])) if state['nodes'][idx]['id'] in cloth_in_washingmachine]
        for i in cloth_in_washingmachine_cond:
            state['nodes'][i]['states'].append("WASHED")
            nodes_with_additional_states[state['nodes'][i]['id']] = state['nodes'][i]

    faucet_near_sink = [n['from_id'] for n in state["edges"] if  n['from_id'] in faucet_ids] # and n["relation_type"] == "CLOSE"]
    faucet_cond = [state['nodes'][idx]['id'] for idx in range(len(state['nodes'])) if state['nodes'][idx]['id'] in faucet_near_sink and 'ON' in state['nodes'][idx]['states']]
    # print(sink_ids, faucet_ids, faucet_near_sink, faucet_cond)
    if len(faucet_cond)>0:
        # print("faucet on")
        utensil_in_sink = [n['from_id'] for n in state["edges"] if n['to_id'] in sink_ids and n["relation_type"] == "INSIDE"]
        utensil_in_sink_cond = [idx for idx in range(len(state['nodes'])) if state['nodes'][idx]['id'] in utensil_in_sink]
        for i in utensil_in_sink_cond:
            # print("utensit in sink")
            state['nodes'][i]['states'].append("WASHED")
            nodes_with_additional_states[state['nodes'][i]['id']] = state['nodes'][i]

    return nodes_with_additional_states
