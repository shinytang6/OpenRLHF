
import os
import json

def save_rollout_data_for_rl_logging_board(experience, args, tokenizer, global_step: int):
    """
    Saving current log in replay buffer for rl logging board.
    """
    import torch
    os.makedirs(args.use_rl_logging_board, exist_ok=True)

    with open(os.path.join(args.use_rl_logging_board, f"rl_logging_board_data.jsonl"), 'a') as f:
        for sequence, action_log_prob, base_action_log_prob, value, token_reward, rm_reward in zip(
            experience.sequences,
            experience.action_log_probs,
            experience.base_action_log_probs,
            # experience.values,
            experience.values if experience.values is not None else [None] * len(experience.sequences),
            experience.token_rewards,
            experience.info['reward'],
        ):
            prompt_ids = sequence[:-len(action_log_prob)]
            prompt = tokenizer.decode(prompt_ids)
            response = tokenizer.decode(sequence[-len(action_log_prob):])
            response_tokens = [tokenizer.decode(_id) for _id in sequence[-len(action_log_prob):]]
            sample = {
                "prompt": prompt,
                "response": response,
                "response_tokens": response_tokens,
                "logprobs": action_log_prob.tolist(),
                "ref_logprobs": base_action_log_prob.tolist(),
                "values": value.tolist() if value is not None else None,
                "token_rewards": token_reward.tolist(),
                "reward": float(rm_reward),
                "step": global_step
            }
            f.write(f"{json.dumps(sample, ensure_ascii=False)}\n")