# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


"""
This script evaluates plan generation using openAI LLMs
for the VirtualHome environment tasks
"""

import sys
sys.path.append("virtualhome/simulation")
sys.path.append("virtualhome/demo")
sys.path.append("virtualhome")

import argparse
import os
import os.path as osp
import random

from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication
from virtualhome.demo.utils_demo import *

import openai
import json
import time

from utils_execute import *

def eval(final_states, 
         final_states_GT, 
         initial_states, 
         test_tasks,
         exec_per_task,
         log_file):
    
    ## the evaluation criteria is not perfect
    ## since sometimes the tasks are underspecified, like which object to interact with
    ## for example "turn off lightswitch" could happen in multiple locations
    ## the evaluation happens w.r.t one possible valid state
    ## that the annotator provides

    sr = []
    unsatif_conds = []; unchanged_conds = []
    total_goal_conds = []; total_unchanged_conds = []
    results = {}
    for g, g_gt, g_in, d in zip(final_states, final_states_GT, initial_states, test_tasks):
        obj_ids = dict([(node['id'], node['class_name']) for node in g_in['nodes']])
        relations_in = set([obj_ids[n['from_id']] +' '+ n["relation_type"] +' '+ obj_ids[n['to_id']] for n in g_in["edges"]])
        obj_states_in = set([node['class_name'] + ' ' + st for node in g_in['nodes'] for st in node['states']])
        
        obj_ids = dict([(node['id'], node['class_name']) for node in g['nodes']])
        relations = set([obj_ids[n['from_id']] +' '+ n["relation_type"] +' '+ obj_ids[n['to_id']] for n in g["edges"]])
        obj_states = set([node['class_name'] + ' ' + st for node in g['nodes'] for st in node['states']])

        obj_ids = dict([(node['id'], node['class_name']) for node in g_gt['nodes']])
        relations_gt = set([obj_ids[n['from_id']] +' '+ n["relation_type"] +' '+ obj_ids[n['to_id']] for n in g_gt["edges"]])
        obj_states_gt = set([node['class_name'] + ' ' + st for node in g_gt['nodes'] for st in node['states']])

        log_file.write(f"\nunsatisfied state conditions: relations: {(relations_gt - relations_in) - (relations - relations_in)}, object states: {(obj_states_gt - obj_states_in) - (obj_states - obj_states_in)}")
        unsatif_conds.append((len((relations_gt - relations_in) - (relations - relations_in))+len((obj_states_gt - obj_states_in) - (obj_states - obj_states_in))))
        total_goal_conds.append(len(relations_gt - relations_in)+len(obj_states_gt - obj_states_in))
        sr.append(1-unsatif_conds[-1]/total_goal_conds[-1])

        unchanged_conds.append((len(relations_gt.intersection(relations_in) - relations)+len(obj_states_gt.intersection(obj_states_in) - obj_states)))
        total_unchanged_conds.append(len(relations_gt.intersection(relations_in))+len(obj_states_gt.intersection(obj_states_in)))
        
        results[d] = {'PSR': sr[-1],
                        "SR": sr[-1:].count(1.0),
                        "Precision": 1-unchanged_conds[-1]/total_unchanged_conds[-1],
                        "Exec": exec_per_task[-1]
                        }

    
    results["overall"] = {'PSR': sum(sr)/len(sr),
                            "SR": sr.count(1.0)/len(sr),
                            "Precision": 1-sum(unchanged_conds)/sum(total_unchanged_conds),
                            "Exec": sum(exec_per_task)/len(exec_per_task)
                            }
    return results

def planner_executer(args):

    # initialize env
    comm = UnityCommunication(file_name=args.unity_filename, 
                              port=args.port, 
                              x_display=args.display)
    
    # prompt example environment is set to env_id 0
    comm.reset(0)

    _, env_graph = comm.environment_graph()
    obj = list(set([node['class_name'] for node in env_graph["nodes"]]))

    # define available actions and append avaailable objects from the env
    prompt = f"from actions import turnright, turnleft, walkforward, walktowards <obj>, walk <obj>, run <obj>, grab <obj>, switchon <obj>, switchoff <obj>, open <obj>, close <obj>, lookat <obj>, sit <obj>, standup, find <obj>, turnto <obj>, drink <obj>, pointat <obj>, watch <obj>, putin <obj> <obj>, putback <obj> <obj>"
    prompt += f"\n\nobjects = {obj}"

    # load train split for task examples
    with open(f"{args.progprompt_path}/data/pythonic_plans/train_complete_plan_set.json", 'r') as f:
        tmp = json.load(f)
        prompt_egs = {}
        for k, v in tmp.items():
            prompt_egs[k] = v
    # print("Loaded %d task example" % len(prompt_egs.keys()))

    ## define the prompt example task setting ##

    # default examples from the paper
    if args.prompt_task_examples == "default":
        default_examples = ["put_the_wine_glass_in_the_kitchen_cabinet",
                            "throw_away_the_lime",
                            "wash_mug",
                            "refrigerate_the_salmon",
                            "bring_me_some_fruit",
                            "wash_clothes",
                            "put_apple_in_fridge"]
        for i in range(args.prompt_num_examples):
            prompt += "\n\n" + prompt_egs[default_examples[i]]

    # random egs - change seeds
    if args.prompt_task_examples == "random":
        random.seed(args.seed)
        prompt_egs_keys = random.sample(list(prompt_egs.keys()), args.prompt_num_examples)

        for eg in prompt_egs_keys:
            prompt += "\n\n" + prompt_egs[eg]

    # abalation settings
    if args.prompt_task_examples_ablation == "no_comments":
        prompt = prompt.split('\n')
        prompt = [line for line in prompt if "# " not in line]
        prompt  = "\n".join(prompt)

    if args.prompt_task_examples_ablation == "no_feedback":
        prompt = prompt.split('\n')
        prompt = [line for line in prompt if not any([x in line for x in ["assert", "else"]])]
        prompt  = "\n".join(prompt)

    if args.prompt_task_examples_ablation == "no_comments_feedback":
        prompt = prompt.split('\n')
        prompt = [line for line in prompt if not any([x in line for x in ["assert", "else", "# "]])]
        prompt  = "\n".join(prompt)


    # evaluate in given unseen env
    if args.env_id != 0:
        comm.reset(args.env_id)
        _, graph = comm.environment_graph()
        obj = list(set([node['class_name'] for node in graph["nodes"]]))
        prompt += f"\n\n\nobjects = {obj}"

        # evaluation tasks in given unseen env
        test_tasks = []
        with open(f"{args.progprompt_path}/data/new_env/{args.test_set}_annotated.json", 'r') as f:
            for line in f.readlines():
                test_tasks.append(list(json.loads(line).keys())[0])
        log_file.write(f"\n----Test set tasks----\n{test_tasks}\nTotal: {len(test_tasks)} tasks\n")

    # setup logging
    log_filename = f"{args.expt_name}_{args.prompt_task_examples}_{args.prompt_num_examples}examples"
    if args.prompt_task_examples_ablation != "none":
        log_filename += f"_{args.prompt_task_examples_ablation}"
    log_filename += f"_{args.test_set}"
    log_file = open(f"{args.progprompt_path}/results/{log_filename}_logs.txt", 'w')
    log_file.write(f"\n----PROMPT for planning----\n{prompt}\n")
    
    # evaluate in seen env
    if args.env_id == 0:
        test_tasks = []
        for file in os.listdir(f"{args.progprompt_path}/data/{args.test_set}"):
            with open(f"{args.progprompt_path}/data/{args.test_set}/{file}", 'r') as f:
                for line in f.readlines():
                    test_tasks.append(list(json.loads(line).keys())[0])

        log_file.write(f"\n----Test set tasks----\n{test_tasks}\nTotal: {len(test_tasks)} tasks\n")

    # test_tasks = test_tasks[:3] ## debug to check changes

    # generate plans for the test set
    if not args.load_generated_plans:
        gen_plan = []
        for task in test_tasks:
            print(f"Generating plan for: {task}\n")
            prompt_task = "def {fxn}():".format(fxn = '_'.join(task.split(' ')))
            curr_prompt = f"{prompt}\n\n{prompt_task}\n\t"
            _, text = LM(curr_prompt, 
                        args.gpt_version, 
                        max_tokens=600, 
                        stop=["def"], 
                        frequency_penalty=0.15)
            gen_plan.append(text)
            # because codex has query limit per min
            if args.gpt_version == 'code-davinci-002':
                time.sleep(90)

        # save generated plan
        line = {}
        print(f"Saving generated plan at: {log_filename}_plans.json\n")
        with open(f"{args.progprompt_path}/results/{log_filename}_plans.json", 'w') as f:
            for plan, task in zip(gen_plan, test_tasks):
                line[task] = plan
            json.dump(line, f)

    # load from file
    else:
        print(f"Loading generated plan from: {log_filename}.json\n")
        with open(f"{args.progprompt_path}/results/{log_filename}_plans.json", 'r') as f:
            data = json.load(f)
            test_tasks, gen_plan = [], []
            for k, v in data.items():
                test_tasks.append(k)
                gen_plan.append(v)

    
    log_file.write(f"\n----PROMPT for state check----\n{current_state_prompt}\n")

    # run execution
    print(f"\n----Runing execution----\n")
    final_states, initial_states, exec_per_task = run_execution(args, 
                                                                comm, 
                                                                test_tasks, 
                                                                gen_plan,
                                                                log_file)
    

    #evaluate
    final_states_GT = []
    with open(f'{args.progprompt_path}/data/final_states/final_states_{args.test_set}.json', 'r') as f:
        for line in f.readlines():
            final_states_GT.append((json.loads(line)))

    results = eval(final_states, 
         final_states_GT, 
         initial_states, 
         test_tasks,
         exec_per_task,
         log_file)

    print(f"\n----Results----\n{results['overall']}\n")
    with open(f"{args.progprompt_path}/results/{log_filename}_metric.json", 'w') as f:
        json.dump(results, f)
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--progprompt-path", type=str, required=True)
    parser.add_argument("--expt-name", type=str, required=True)

    parser.add_argument("--openai-api-key", type=str, 
                        default="sk-xyz")
    parser.add_argument("--unity-filename", type=str, 
                        default="/path/to/macos_exec.v2.3.0.app")
    parser.add_argument("--port", type=str, default="8000")
    parser.add_argument("--display", type=str, default="0")
    
    parser.add_argument("--gpt-version", type=str, default="text-davinci-002", 
                        choices=['text-davinci-002', 'davinci', 'code-davinci-002'])
    parser.add_argument("--env-id", type=int, default=0)
    parser.add_argument("--test-set", type=str, default="test_unseen", 
                        choices=['test_unseen', 'test_seen', 'test_unseen_ambiguous', 'env1', 'env2'])

    parser.add_argument("--prompt-task-examples", type=str, default="default", 
                        choices=['default', 'random'])
    # for random task examples, choose seed
    parser.add_argument("--seed", type=int, default=0)
    
    ## NOTE: davinci or older GPT3 versions have a lower token length limit
    ## check token length limit for models to set prompt size: 
    ## https://platform.openai.com/docs/models
    parser.add_argument("--prompt-num-examples", type=int, default=3, 
                         choices=range(1, 7))
    parser.add_argument("--prompt-task-examples-ablation", type=str, default="none", 
                         choices=['none', 'no_comments', "no_feedback", "no_comments_feedback"])

    parser.add_argument("--load-generated-plans", type=bool, default=False)
    
    args = parser.parse_args()
    openai.api_key = args.openai_api_key

    if not osp.isdir(f"{args.progprompt_path}/results/"):
            os.makedirs(f"{args.progprompt_path}/results/")

    planner_executer(args=args)