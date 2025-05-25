import asyncio
import os
from collections import Counter
from typing import Any

import pandas as pd
from commit0.harness.constants import SPLIT
from datasets import load_dataset

import openhands.agenthub
from evaluation.utils.shared import (
    EvalException,
    EvalMetadata,
    EvalOutput,
    assert_and_raise,
    codeact_user_response,
    get_default_sandbox_config_for_eval,
    make_metadata,
    prepare_dataset,
    reset_logger_for_multiprocessing,
    run_evaluation,
    update_llm_config_for_completions_logging,
)
from openhands.controller.state.state import State
from openhands.core.config import (
    AgentConfig,
    AppConfig,
    get_llm_config_arg,
    get_parser,
)
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.events.action import CmdRunAction, MessageAction
from openhands.events.observation import CmdOutputObservation, ErrorObservation
from openhands.events.serialization.event import event_to_dict
from openhands.runtime.base import Runtime
from openhands.utils.async_utils import call_async_from_sync
from openhands.utils.shutdown_listener import sleep_if_should_continue

USE_HINT_TEXT = os.environ.get('USE_HINT_TEXT', 'false').lower() == 'true'
RUN_WITH_BROWSING = os.environ.get('RUN_WITH_BROWSING', 'false').lower() == 'true'

AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
    'CodeActAgent': codeact_user_response,
    'CodeActCommit0Agent': codeact_user_response,
}

def get_instruction(instance: pd.Series, metadata: EvalMetadata):
    repo_name = instance['repo'].split('/')[1]
    # Instruction based on Anthropic's official trajectory
    # https://github.com/eschluntz/swe-bench-experiments/tree/main/evaluation/verified/20241022_tools_claude-3-5-sonnet-updated/trajs
    instruction = (
        # f'''Build a simple app'''
        f'''
        Using the README.md in the repo:
        1. Set up the environment
        2. Set up dependencies
        3. Build the project
        Project Name: {repo_name}
        Target OS: Linux4
        Note: Do not run the project
        '''
    )

    if RUN_WITH_BROWSING:
        instruction += (
            '<IMPORTANT!>\n'
            'You SHOULD NEVER attempt to browse the web. '
            '</IMPORTANT!>\n'
        )
    return instruction


# # TODO: migrate all swe-bench docker to ghcr.io/openhands
# DOCKER_IMAGE_PREFIX = os.environ.get(
#     'EVAL_DOCKER_IMAGE_PREFIX', 'docker.io/wentingzhao/'
# )
# logger.info(f'Using docker image prefix: {DOCKER_IMAGE_PREFIX}')


# def get_instance_docker_image(repo_name: str) -> str:
#     return (DOCKER_IMAGE_PREFIX.rstrip('/') + '/' + repo_name).lower() + ':v0'


def get_config(
    instance: pd.Series,
    metadata: EvalMetadata,
) -> AppConfig:
    repo_name = instance['repo'].split('/')[1]
    #base_container_image = get_instance_docker_image(repo_name)
    # logger.info(
    #     f'Using instance container image: {base_container_image}. '
    #     f'Please make sure this image exists. '
    #     f'Submit an issue on https://github.com/All-Hands-AI/OpenHands if you run into any issues.'
    # )

    sandbox_config = get_default_sandbox_config_for_eval()
    #sandbox_config.base_container_image = base_container_image

    ###### Check out SWEBENCH code for docker

    config = AppConfig(
        default_agent=metadata.agent_class,
        run_as_openhands=False,
        max_iterations=metadata.max_iterations,
        runtime=os.environ.get('RUNTIME', 'docker'),
        sandbox=sandbox_config,
        # do not mount workspace
        workspace_base=None,
        workspace_mount_path=None,
    )
    config.set_llm_config(
        update_llm_config_for_completions_logging(
            metadata.llm_config, metadata.eval_output_dir, instance['instance_id']
        )
    )
    agent_config = AgentConfig(
        enable_jupyter=False,
        enable_browsing=RUN_WITH_BROWSING,
        enable_llm_editor=False,
    )
    config.set_agent_config(agent_config)
    return config


def initialize_runtime(
    runtime: Runtime,
    instance: pd.Series,  # this argument is not required
):
    """Initialize the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    """
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Initialization Fn')
    logger.info('-' * 30)
    workspace_dir_name = instance['repo'].split('/')[1]
    obs: CmdOutputObservation

    action = CmdRunAction(
        command=f'git clone https://github.com/{instance["repo"]}.git'
    )
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        obs.exit_code == 0,
        f'Failed to git clone https://github.com/{instance["repo"]}.git: {str(obs)}',
    )

    action = CmdRunAction(command=f'cd {workspace_dir_name}')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        obs.exit_code == 0,
        f'Failed to cd to /{workspace_dir_name}: {str(obs)}',
    )

    logger.info('-' * 30)
    logger.info('END Runtime Initialization Fn')
    logger.info('-' * 30)


def complete_runtime(
    runtime: Runtime,
    instance: pd.Series,  # this argument is not required, but it is used to get the workspace_dir_name
) -> dict[str, Any]:
    """Complete the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    If you need to do something in the sandbox to get the correctness metric after
    the agent has run, modify this function.
    """
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Completion Fn')
    logger.info('-' * 30)
    obs: CmdOutputObservation

    # except json.JSONDecodeError:
    #     logger.error('Failed to parse test report JSON')
    #     eval_result = {
    #         'name': workspace_dir_name,
    #         'sum': 0,
    #         'passed': 0,
    #         'num_passed': 0,
    #         'num_tests': len(test_ids),ß
    #     }

    # # Create tarball of workspace
    # temp_zip = runtime.copy_from(f'/workspace/{workspace_dir_name}')

    # commit0_dir = os.path.dirname(__file__)
    # persistent_zip = os.path.join(commit0_dir, f'{workspace_dir_name}.zip')
    # with open(temp_zip, 'rb') as src, open(persistent_zip, 'wb') as dst:
    #     dst.write(src.read())
    # zip_file = persistent_zip

    logger.info('-' * 30)
    logger.info('END Runtime Completion Fn')
    logger.info('-' * 30)

    return {
        'eval_result': {},
        #'zip_file': zip_file,
    }

    


def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
) -> EvalOutput:
    config = get_config(instance, metadata)
    # Setup the logger properly, so you can run multi-processing to parallelize the evaluation
    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, instance.instance_id, log_dir)
    else:
        logger.info(f'Starting evaluation for instance {instance.instance_id}.')

    runtime = create_runtime(config)
    call_async_from_sync(runtime.connect)
    try:
        initialize_runtime(runtime, instance)

        instruction = get_instruction(instance, metadata)

        # Here's how you can run the agent (similar to the `main` function) and get the final task state
        state: State | None = asyncio.run(
            run_controller(
                config=config,
                initial_user_action=MessageAction(content=instruction),
                runtime=runtime,
                fake_user_response_fn=AGENT_CLS_TO_FAKE_USER_RESPONSE_FN[
                    metadata.agent_class
                ],
            )
        )

        # if fatal error, throw EvalError to trigger re-run
        if (
            state.last_error
            and 'fatal error during agent execution' in state.last_error
            and 'stuck in a loop' not in state.last_error
        ):
            raise EvalException('Fatal error detected: ' + state.last_error)

        return_val = complete_runtime(runtime, instance)
        eval_result = return_val['eval_result']

        logger.info(
            f'Got evaluation result for repo {instance.instance_id}:\n--------\n{eval_result}\n--------'
        )
    finally:
        runtime.close()
    # ==========================================

    # ======= Attempt to evaluate the agent's edits =======
    # we use eval_infer.sh to evaluate the agent's edits, not here
    # because the agent may alter the environment / testcases
    test_result = {
        'eval_result': eval_result,
    }
 
    # If you are working on some simpler benchmark that only evaluates the final model output (e.g., in a MessageAction)
    # You can simply get the LAST `MessageAction` from the returned `state.history` and parse it for evaluation.
    if state is None:
        raise ValueError('State should not be None.')

    # NOTE: this is NO LONGER the event stream, but an agent history that includes delegate agent's events
    histories = [event_to_dict(event) for event in state.history]
    metrics = state.metrics.get() if state.metrics else None

    # Save the output
    output = EvalOutput(
        instance_id=instance.instance_id,
        instruction=instruction,
        instance=instance.to_dict(),
        test_result=test_result,
        metadata=metadata,
        history=histories,
        metrics=metrics,
        error=state.last_error if state and state.last_error else None,
    )
    return output


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='islamham/test-project',
        help='dataset to evaluate on, only test split exists for this HF dataset',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        help='this is the HF dataset split',
    )

    args, _ = parser.parse_known_args()

    # NOTE: It is preferable to load datasets from huggingface datasets and perform post-processing
    # so we don't need to manage file uploading to OpenHands's repo
    dataset = load_dataset(args.dataset, split=args.split)

    llm_config = None
    if args.llm_config:
        print(args.llm_config)
        llm_config = get_llm_config_arg(args.llm_config)
        print(llm_config)
        # modify_params must be False for evaluation purpose, for reproducibility and accurancy of results
        llm_config.modify_params = False
        llm_config.log_completions = True

    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    details = {}
    _agent_cls = openhands.agenthub.Agent.get_cls(args.agent_cls)

    dataset_descrption = (
        args.dataset.replace('/', '__')
    )
    metadata = make_metadata(
        llm_config,
        dataset_descrption,
        args.agent_cls,
        args.max_iterations,
        args.eval_note,
        args.eval_output_dir,
        details=details,
    )

    output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')

    instances = prepare_dataset(dataset.to_pandas(), output_file, args.eval_n_limit)

    run_evaluation(
        instances,
        metadata,
        output_file,
        args.eval_num_workers,
        process_instance,
        timeout_seconds=10*60,
        max_retries=1
    )
