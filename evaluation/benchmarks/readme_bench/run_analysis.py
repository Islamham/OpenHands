import json
import os
import pprint
from datetime import datetime

from openhands.core.config import get_llm_config_arg, get_parser, load_app_config
from openhands.core.logger import openhands_logger as logger
from openhands.llm.llm import LLM

config = load_app_config()

build_tools = ["python3", "python", "pip", "poetry", "mvn", "mvnw", "gradle", "gradlew", "setup.py", "setuptools"]
build_commands = ["build", "install", "validate", "compile", "test", "package", "integration-test", "verify", "deploy", "assemble", "init", "wrapper", "dependencies", "resolve", "publish", "installDist", "buildPlugin"]

def extract_metrics_from_output(path: str):

    with open(path, 'r') as output:
        for line in output:
            data = json.loads(line.strip())
            instance_id = data['instance_id']
            if instance_id in instance_ids:
                print(f'WARNING: Duplicate instance_id found: {instance_id}')
                continue
            instance_ids.add(instance_id)

            history = data.get('history')
            if not history:
                analysis.append({
                        'instance_id': instance_id,
                        'error': data['error']
                    }
                )
                continue
            else:
                first, last = history[0], history[-1]
                runtime_in_seconds = (datetime.fromisoformat(last['timestamp']) - 
                                   datetime.fromisoformat(first['timestamp'])).total_seconds()
                action_counts = {
                    'read': 0, 'edit': 0, 'think': 0,
                    'message': 0, 'run': 0, 'call_tool_mcp': 0
                }
                thought = None
                build_instructions_exist = True
                subtract_runtime = 0
                subtract_prompt_tokens = 0
                subtract_completion_tokens = 0
                subtract_cost = 0
                subtract_num_agent_actions = 0
                for event in history:
                    if event['source'] == 'agent' and 'action' in event:
                        action = event['action']
                        if action in action_counts:
                            action_counts[action] += 1
                        elif action not in ['finish', 'system']:
                            logger.warning(f'unknown agent action: {action} on Instance {instance_id}')
                        thought = event['args'].get('thought', '')
                        if build_instructions_exist:
                            if all(word in thought for word in ['does not contain', 'build', 'instructions']):
                                build_instructions_exist = False
                        if not any(tool in event['args']['command'] for tool in build_tools):
                            if not any(command in event['args']['command'] for command in build_commands):
                                break
                            else:
                                # for the runtime, get the timestamp for this action id, get the timestamp for the observation next, add the total seconds to a subtract_runtime var
                                # for tokens, get the tokens for the action and add to a subtract tokens var for each type of token
                                # same for the cost
                                # increment subtract_num_agent_actions count
                                subtract_runtime += 1
                                subtract_prompt_tokens += 1
                                subtract_completion_tokens += 1
                                subtract_cost += 1
                                subtract_num_agent_actions += 1
                task_completed = last.get('args', {}).get('task_completed')
                final_thought = last.get('args', {}).get('final_thought')
                metrics = data['metrics']
                accumulated_token_usage = metrics['accumulated_token_usage']
                analysis.append({   
                    'instance_id': instance_id,
                    'repo': data['instance']['repo'],
                    'metrics': {
                        'accumulated_cost': metrics['accumulated_cost'],
                        'accumulated_prompt_token_usage': accumulated_token_usage['prompt_tokens'],
                        'accumulated_completion_token_usage': accumulated_token_usage['completion_tokens'],
                        'runtime_in_seconds': runtime_in_seconds,
                        'total_num_agent_actions':  sum(action_counts.values()),
                        'accumulated_cost_no_build': metrics['accumulated_cost'] - subtract_cost,
                        'accumulated_prompt_token_usage_no_build': accumulated_token_usage['prompt_tokens'] - subtract_prompt_tokens,
                        'accumulated_completion_token_usage_no_build': accumulated_token_usage['completion_tokens'] - subtract_completion_tokens,
                        'runtime_in_seconds_no_build': runtime_in_seconds - subtract_runtime,
                        'total_num_agent_actions_no_build':  sum(action_counts.values()) - subtract_num_agent_actions,
                        
                    },
                    'build_instructions_exist': build_instructions_exist,
                    'task_completed': task_completed,
                    'thought': thought if not final_thought else None,
                    'final_thought': final_thought,
                    'error': data['error'],
                })
            # except Exception as e:
            #     logger.warning(f'could not get metrics from output, Instance {instance_id} had error: {e}')
                

def extract_timed_out_ids(path: str) -> list:
    with open(path, 'r') as output:
        ids = []
        for line in output:
            data = json.loads(line.strip())
            if data.get('error') and 'Timeout' in data['error']:
                ids.append(data['instance_id'])
        return ids
    
def extract_max_iter_ids(path: str) -> list:
    with open(path, 'r') as output:
        ids = []
        for line in output:
            data = json.loads(line.strip())
            if data.get('error') and 'maximum iteration' in data['error']:
                ids.append(data['instance_id'])
        return ids

def extract_task_failed_ids(path: str) -> list:
    with open(path, 'r') as output:
        ids = []
        for line in output:
            data = json.loads(line.strip())
            history = data.get('history')
            if not history:
                continue
            last = history[-1]
            task_completed = last.get('args', {}).get('task_completed')
            if task_completed == 'false':
                ids.append(data['instance_id'])
        return ids


# def classify_error(llm: LLM, failed_case: dict) -> str:

#     prompt = f"""
#     Please classify the error for the following failed case based on the history and eval_output:

#     Instruction:
#     {failed_case['instruction']}

#     Eval Script:
#     {failed_case['eval_script']}s

#     History:
#     {failed_case['history']}

#     Eval Output:
#     {failed_case['eval_output']}

#     The error categories are:
#     E1: Hallucination Errors - The model misinterpreted the user's intention, misplaced Python code and bash script, or generated random or irrelevant code.
#     E2: Lack of Knowledge or Information - The model lacks sufficient information or domain-specific knowledge to satisfy the user's requirements.
#     E3: Knowledge Manipulation - The model failed to integrate or manipulate information properly.
#     E4: Syntax Errors - The model generated code with syntax errors.
#     E5: Operational Error - The model gave up easily or exited without finishing the tasks.

#     Please provide only the error category (E1, E2, E3, E4, or E5) without any explanation.
#     """

#     try:
#         response = llm.completion(messages=[{'content': prompt, 'role': 'user'}])
#         error_category = response.choices[0].message['content']
#     except Exception as e:
#         logger.error(
#             f'Failed to classify the error for the failed case: {failed_case["instance_id"]}'
#         )
#         logger.error(e)
#         error_category = input(
#             failed_case['instruction']
#             + ': '
#             + failed_case['eval_script']
#             + ' - '
#             + failed_case['eval_output']
#         )

#     if error_category not in ['E1', 'E2', 'E3', 'E4', 'E5']:
#         raise ValueError(f'Invalid error category: {error_category}')

#     return error_category


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='dataset to extract the evaluation results for',
    )
    args, _ = parser.parse_known_args()

    dataset = args.dataset
    print(dataset)
    dataset_eval_path = ''

    for (root, _, _) in os.walk(args.eval_output_dir):
        if str(dataset) in str(root):
            dataset_eval_path = root
            break

    runs = {}

    for (root, subdir, _) in os.walk(dataset_eval_path):
        if 'CodeActAgent' in root:
            for file in subdir:
                runs[file] = os.path.join(root,file)
            break

    if not os.path.exists(dataset_eval_path):
        raise ValueError(f'Dataset evaluation path not found: {dataset_eval_path}')

    for i, (name, path) in enumerate(runs.items(), 1):
        output_file_path = os.path.join(path, "output.jsonl")
        analysis = []
        instance_ids = set()
        
        logger.info(f'Starting analysis for {dataset} runs [{i}/{len(runs)}]:\n{name}')
        extract_metrics_from_output(output_file_path)

        analysis.sort(key=lambda d: int(d['instance_id']))
        output_name = f'analysis_{dataset}_{datetime.now()}_{name}.jsonl'
        
        with open(output_name, 'w') as output:
            for instance in analysis:  
                output.write(json.dumps(instance) + '\n')
         
        # Extract different types of IDs
        id_types = {
            'timed_out': extract_timed_out_ids(output_file_path),
            'max_iter': extract_max_iter_ids(output_file_path),
            'task_failed': extract_task_failed_ids(output_file_path)
        }

        # Save all ID types
        for id_type, ids in id_types.items():
            output_name = f'{id_type}_ids_{dataset}_{datetime.now()}_{name}.jsonl'
            with open(output_name, 'w') as output:
                output.write(json.dumps(ids) + '\n')

