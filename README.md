# ProgPrompt on VirtualHome

This is the code release for the paper [ProgPrompt: Generating Situated Robot Task Plans using Large Language Models](https://progprompt.github.io/). It contains code for replicating the results on the VirtualHome dataset.



## Setup

Create a conda environment (or your virtualenv):
```
conda create -n progprompt python==3.9
```

Install dependencies:
```
pip install -r requirements.txt
```

Clone [VirtualHome](https://github.com/xavierpuigf/virtualhome) and install it by running:
```
pip install -e .
```

**Note:** If you an encounter an error to do with wrong number of arguments to function `execute`, then in file `virtualhome/src/virtualhome/simulation/evolving_graph/execution.py` line 67, add `*args` as follows:
```
    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo, char_index, *args):
```
This was tested on VirtualHome commit `f84ee28a75b23318ee1bf652862b1c993269cd06`.

Finally, download the virtualhome unity simulator and make sure it runs. The simulator can run on the desktop, or on a virtual x-server.


## Running evaluation

Here is a minimal example how to run the evaluation script. Replace {arguments in curly braces} with appropriate values on your system:
```
python3 scripts/run_eval.py --progprompt-path $(pwd) --expt-name {expt_name} --openai-api-key {key} --unity-filename {v2.3_virtualhome_sim} --display {0}
```

For more options and arguments, look inside `scripts/run_eval.py`. 
