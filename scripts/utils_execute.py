# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import random
from virtualhome.demo.utils_demo import *

import openai
import numpy as np
from virtualhome.simulation.evolving_graph import utils
from virtualhome.simulation.evolving_graph.scripts import parse_script_line, Script
from virtualhome.simulation.evolving_graph.execution import ScriptExecutor
from virtualhome.simulation.evolving_graph.environment import EnvironmentGraph
import time
import re

from utils_aug_env import get_obj_ids_for_adding_states, add_additional_obj_states

def LM(prompt, 
       gpt_version,
       max_tokens=128, 
       temperature=0, 
       stop=None, 
       logprobs=1, 
       frequency_penalty=0):
    
    ## function to query LM ##
    # you may adjust the genration parameters as needed
    # more info on parameters here: 
    # https://platform.openai.com/docs/api-reference/completions/create
    response = openai.Completion.create(model=gpt_version, 
                                        prompt=prompt, 
                                        max_tokens=max_tokens, 
                                        temperature=temperature, 
                                        stop=stop, 
                                        logprobs=logprobs, 
                                        frequency_penalty = frequency_penalty)

    return response, response["choices"][0]["text"].strip()

def get_current_state_prompt():
    ## fixed function to define "PROMPT for state check"
    current_state_prompt = "kitchencounterdrawer, door is OPEN, character, wallpictureframe, clothespile is CLOSED, coffeemaker is OFF, pie, wall, bedroom, microwave is OFF and CLOSED, lightswitch is ON, kitchencabinet is CLOSED, washingsponge, bellpepper, salmon, fridge is CLOSED, wallshelf, tvstand, paper, floor, chips, photoframe, kitchen, whippedcream, candybar, faucet is OFF, tv is OFF, cereal, stovefan, waterglass, cutleryknife, kitchentable, condimentbottle, wineglass, bookshelf, cutleryfork, chocolatesyrup, walllamp, bench, sink, crackers, orchid, condimentshaker, kitchencounter is CLOSED, livingroom, powersocket, coffeepot is CLOSED, creamybuns, ceilinglamp, rug, book is CLOSED, plate, toaster is OFF, clock is OFF, wallphone is OFF, ceiling, fryingpan, box is CLOSED, dishbowl, bananas, breadslice, bathroom, garbagecan is CLOSED, stove is OFF and CLOSED, dishwashingliquid, plate ON kitchencounter, cutleryfork ON kitchentable, bookshelf ON floor, cutleryknife ON kitchentable, bellpepper ON kitchencounter, microwave ON kitchencounterdrawer, chocolatesyrup ON wallshelf, whippedcream ON rug, salmon ON microwave, orchid ON tvstand, wallpictureframe ON wall, bench ON floor, tvstand ON floor, book INSIDE bookshelf, bananas ON dishbowl, toaster ON kitchencounterdrawer, whippedcream ON kitchentable, dishbowl INSIDE bookshelf, fryingpan ON stove, rug ON kitchentable, coffeepot INSIDE coffeemaker, waterglass ON rug, dishwashingliquid ON kitchencounter, wallshelf ON wall, washingsponge ON kitchencounter, clothespile INSIDE bookshelf, bananas INSIDE bookshelf, box ON bookshelf, plate ON kitchentable, waterglass ON kitchentable, creamybuns ON wallshelf, breadslice INSIDE toaster, coffeemaker ON kitchencounterdrawer, chips ON wallshelf, book ON kitchentable, dishbowl ON bookshelf, pie ON kitchentable, wineglass ON tvstand, box ON tvstand, coffeepot ON kitchencounter, bellpepper ON kitchencounterdrawer, condimentshaker INSIDE bookshelf, coffeemaker ON kitchencounter, toaster ON kitchencounter, box INSIDE bookshelf, crackers ON wallshelf, character HOLD_RH book, faucet ON kitchencounter, book ON rug, cereal ON wallshelf, plate INSIDE microwave, candybar ON wallshelf, condimentbottle INSIDE bookshelf, tv ON tvstand, microwave ON kitchencounter, paper INSIDE bookshelf, kitchencounterdrawer ON kitchencounter, fridge ON floor, photoframe ON tvstand, wallpictureframe ON wallpictureframe, bench ON rug, pie ON rug, kitchencounterdrawer ON kitchencounterdrawer, dishbowl ON kitchencounter.\n\nassert('close' to 'mug' )\nFalse\nassert('close' to 'microwave' )\nTrue\nassert('book' is 'closed' )\nTrue\nassert('lightswitch' is 'OFF')\nFalse\nassert('book' in 'bookshelf')\nTrue\nassert('book' in 'hands')\nTrue\nassert('cereal' on 'bookshelf')\nFalse"
    objs  = ['microwave', 'book', 'lightswitch', 'bookshelf', 'cereal']
    state, asserts = current_state_prompt = current_state_prompt.split('\n\n')
    state = state.split(',')
    state = "You see: " +  ', '.join([i.strip() for i in state if any(element in i for element in objs)])
    current_state_prompt = f"{state}\n\n{asserts}"
    return current_state_prompt

current_state_prompt = get_current_state_prompt()

def run_execution(args, comm, test_tasks, gen_plan, log_file):
    final_states = []; initial_states = []; exec_per_task = []

    for task, plan in zip(test_tasks, gen_plan):
        ## initialize and set up enviroenment: visual + graph environment ##
        comm.reset(args.env_id)
        comm.add_character('Chars/Male2', initial_room='kitchen')

        _, graph = comm.environment_graph()
        _, cc = comm.camera_count()
        initial_states.append(graph)

        env_graph = EnvironmentGraph(graph)
        name_equivalence = utils.load_name_equivalence()
        executor = ScriptExecutor(env_graph, name_equivalence)

        ## get agent's initial state ##
        agent = [n['id'] for n in graph["nodes"] if n['class_name'] == 'character'][0]
        agent_in_roomid = [n['to_id'] for n in graph["edges"] if n['from_id'] == agent and n["relation_type"] == "INSIDE"][0]
        agent_in_room = [n['class_name'] for n in graph["nodes"] if n['id'] == agent_in_roomid][0]
        agent_has_objid = [n['to_id'] for n in graph["edges"] if n['from_id'] == agent and "HOLD" in n["relation_type"]]
        agent_has_obj = [n['class_name'] for n in graph["nodes"] if n['id'] in agent_has_objid]
        # some actions might not execute in the visual simulation, but they will in evolving graphs
        images = []
        max_fails = 10; num_fails = 0
        _, im = comm.camera_image([cc-5], image_width=300, image_height=300)
        images.append(im[0])
        # s, obj = comm.get_visible_objects(cc-6)
        obj_ids_for_adding_states = get_obj_ids_for_adding_states(graph)
        nodes_with_additional_states = {}

        partial_graph = utils.get_visible_nodes(graph, agent_id=agent)

        obj_ids_close = [n['to_id'] for n in graph["edges"] if n['from_id'] == agent and  n["relation_type"]=="CLOSE"]
        obj = [node['class_name'] for node in partial_graph['nodes'] if node["id"] in obj_ids_close]
        obj_ids = dict([(node['id'], node['class_name']) for node in graph['nodes'] if node["id"] in obj_ids_close and node['class_name'] in obj])
        relations = list(set([obj_ids[n['from_id']] +' '+ n["relation_type"] +' '+ obj_ids[n['to_id']] for n in graph["edges"] if n['from_id'] in obj_ids and n['to_id'] in obj_ids and n["relation_type"] not in ["CLOSE","FACING", "INSIDE"]]))    
        obj_states = [(node['class_name'], node['states']) for node in graph['nodes'] if node['class_name'] in obj]
        objs = ""

        for ob_states in obj_states:
            if len(ob_states[1])>0:
                objs = objs + ob_states[0] + ' is ' + ' and '.join(ob_states[1]) + ', '
            else:
                objs = objs + ob_states[0] + ', '
        objs = list(set(objs.split(', ')))
        objs = [ob for ob in objs if len(ob)>0]
        objs = ', '.join(objs) + ', ' + ', '.join(relations)  + '. '
        if len(agent_has_obj)>0:
            agent_has_obj = ', '.join(agent_has_obj)
            objs += f" You have {agent_has_obj}. "


        ## parse plan into subgoals ##
        log_file.write(f"\n--Executing task: {task}--\n")
        log_file.write(f"Plan:  {plan}\n\n")
        print(f"Executing: {task}\n")

        subgoals = {}
        subgoals['0'] = []
        for i in plan.split('\n'):
            i = i.strip()
            if len(i)<1:
                continue
            if "comments" in args.prompt_task_examples_ablation:
                subgoals['0'].append(i)
            else:
                if "#" in i:
                    sg = i.split("#")[1]; sg = sg.strip()
                    subgoals[sg] = []
                else:
                    subgoals[sg].append(i)

        ## begin execution ##
        executable_steps = 0; total_steps = 0
        last_assert = None
        for subgoal in subgoals.keys():
            step = 1; act = ""
            for action in subgoals[subgoal]:
                # fixes needed for not getting stuck
                if step > 10:
                    break
                if "grab('wallphone')" in action:
                    continue
                
                ## state checking ##

                # parse asserts and query LLM
                if "assert" in action:
                    check_state = ""; last_assert = action
                    assert_objs = re.findall(r"\b[a-z]+", action)[1::2]
                    state = objs.split(',')
                    state = "You see: " + ', '.join([i.strip() for i in state if any(ele in i for ele in assert_objs)])
                    current_state = f"{current_state_prompt}\n\n{state}\n\n{action}\n"
                    _, check_state = LM(current_state, args.gpt_version, 
                                        max_tokens=2, stop=["\n"])
                    log_file.write(f"State check:\n{state}\n{action}\n{check_state.strip()}\n")
                    continue
                
                # get recovery actions
                if last_assert!=None:
                    if "True" in check_state:
                        # skip revovery if state check is true
                        if "else: " in action:
                            continue 
                    elif "False" in check_state:
                        if "else: " in action:
                            action = action.split(': ')[-1].strip()
                        else:
                            state = objs.split(',')
                            state = "You see: " +  ', '.join([i.strip() for i in state if any(ele in i for ele in assert_objs)])
                            current_state = f"{current_state_prompt}\n\n{state}\n\n{action}\n"
                            _, check_state = LM(current_state, args.gpt_version, 
                                                max_tokens=2, stop=["\n"])
                            log_file.write(f"State check:\n{state}\n{action}\n{check_state.strip()}\n")
                    
                # since above steps are not for env, following line go through the env
                total_steps+=1
                
                ## parse next action
                action = action.split(')')[0]
                action = re.findall(r"\b[a-z]+", action)

                if len(action)==3 and "put" in action[0]: # 2 objs action
                    obj_id1 = [node['id'] for node in graph['nodes'] if node['class_name'] == action[1] and node['id'] in agent_has_objid]
                    obj_id2 = [node['id'] for node in graph['nodes'] if node['class_name'] == action[2]]
                    if len(obj_id1)==0:
                        step+1; log_file.write("obj not in hand\n"); continue
                    if len(obj_id1)==1:
                        id1 = obj_id1[0]
                    else:
                        id1 = random.choice(obj_id1)
                    
                    if len(obj_id2)==0:
                        step+1; log_file.write("obj not found\n"); continue
                    elif len(obj_id2)==1:
                        id2 = obj_id2[0]
                    elif found_id in obj_id2:
                        id2 = found_id
                    else:
                        id2 = random.choice(obj_id2)
                    script_instruction = '<char0> [{}] <{}> ({}) <{}> ({})'.format(action[0], action[1], id1, action[2], id2)
                elif len(action)==2 and action[0] not in ["find", "walk"]: # 1 obj action
                    obj_id1 = [node['id'] for node in graph['nodes'] if node['class_name'] == action[1]]
                    if len(obj_id1)==1:
                        id1 = obj_id1[0]
                    elif found_id in obj_id1:
                        id1 = found_id
                    elif len(obj_id1)==0:
                        step+1; log_file.write("obj not found\n"); continue
                    else:
                        id1 = random.choice(obj_id1)
                    script_instruction = '<char0> [{}] <{}> ({})'.format(action[0], action[1], id1)
                elif len(action)==2: # walk or find action
                    obj_id1 = [node['id'] for node in graph['nodes'] if node['class_name'] == action[1]]
                    if len(obj_id1)==0:
                        step+1; log_file.write("obj not found\n"); continue
                    found_id = random.choice(obj_id1)
                    script_instruction = '<char0> [{}] <{}> ({})'.format(action[0], action[1], found_id)
                elif len(action)==1: # 0 object action
                    script_instruction = '<char0> [{}]'.format(action[0])
                else:
                    log_file.write("bad action\n"); continue
                
                ## execute next action in both envs: visual and graph
                log_file.write(f"{script_instruction}\n")
                _, m = comm.render_script([script_instruction], recording=False, skip_animation=True, find_solution=True)
                script = script_instruction[7:]
                try:
                    script = parse_script_line(script, 0)
                except:
                    step+=1; continue
                print(script)
                success, final_state, _ = executor.execute(Script([script])) 
                
                if not success:
                    log_file.write(f"act_success: {success}, message: {executor.info.get_error_string()}\n")
                    step+=1
                else:
                    # count execution if action executes succesfully in graph env
                    executable_steps+=1
                    # _, graph = comm.environment_graph()
                    graph = final_state.to_dict()
                    env_graph = EnvironmentGraph(graph)
                    agent = [n['id'] for n in graph["nodes"] if n['class_name'] == 'character'][0]
                    partial_graph = utils.get_visible_nodes(final_state.to_dict(), agent_id=agent)
                    name_equivalence = utils.load_name_equivalence()
                    executor = ScriptExecutor(env_graph, name_equivalence)
                    script_instruction = ' '.join(re.findall(r"\b[a-z]+", script_instruction)[1:])
                    step+=1

                    # get new state info
                    agent = [n['id'] for n in graph["nodes"] if n['class_name'] == 'character'][0]
                    agent_in_roomid = [n['to_id'] for n in graph["edges"] if n['from_id'] == agent and n["relation_type"] == "INSIDE"][0]
                    agent_in_room = [n['class_name'] for n in graph["nodes"] if n['id'] == agent_in_roomid][0]
                    agent_has_objid = [n['to_id'] for n in graph["edges"] if n['from_id'] == agent and "HOLD" in n["relation_type"]]
                    agent_has_obj = [n['class_name'] for n in graph["nodes"] if n['id'] in agent_has_objid]

                    # Here you can get an observation, for instance 
                    if 'grab' in script_instruction or 'open' in script_instruction or 'close' in script_instruction:
                        s, im = comm.camera_image([cc-5], image_width=300, image_height=300)
                    else:
                        s, im = comm.camera_image([cc-6], image_width=300, image_height=300)
                    images.append(im[0])

                    obj_ids_close = [n['to_id'] for n in graph["edges"] if n['from_id'] == agent and  n["relation_type"]=="CLOSE"]
                    obj = [node['class_name'] for node in partial_graph['nodes'] if node["id"] in obj_ids_close]
                    obj_ids = dict([(node['id'], node['class_name']) for node in partial_graph['nodes'] if node["id"] in obj_ids_close and node['class_name']!=agent_in_room])
                    nodes_with_additional_states = add_additional_obj_states(partial_graph, obj_ids_for_adding_states, nodes_with_additional_states)

                    relations = list(set([obj_ids[n['from_id']] +' '+ n["relation_type"] +' '+ obj_ids[n['to_id']] for n in graph["edges"] if n['from_id'] in obj_ids and n['to_id'] in obj_ids and n["relation_type"]  not in ["CLOSE","FACING"]]))
                    obj_states = [(node['class_name'], node['states']) for node in graph['nodes'] if node['class_name'] in obj]
                    objs = ""
                    for ob_states in obj_states:
                        if len(ob_states[1])>0:
                            objs = objs + ob_states[0] + ' is ' + ' and '.join(ob_states[1]) + ', '
                        else:
                            objs = objs + ob_states[0] + ', '
                    objs = list(set(objs.split(', '))) 
                    objs = [ob for ob in objs if len(ob)>0]
                    objs = ', '.join(objs)+ ', ' + ', '.join(relations) + '. '

                    if len(agent_has_obj)>0:
                        agent_has_obj = ', '.join(agent_has_obj)
                        objs += f" You have {agent_has_obj}. "
                
        # augment state with additional state info           
        final_state = final_state.to_dict()
        for idx in range(len(final_state["nodes"])):
            if final_state["nodes"][idx]['id'] in nodes_with_additional_states.keys():
                final_state["nodes"][idx] = nodes_with_additional_states[final_state["nodes"][idx]['id']]
            
        # get final state for eval
        final_states.append(final_state)
        exec_per_task.append(executable_steps/total_steps)
    return final_states, initial_states, exec_per_task



