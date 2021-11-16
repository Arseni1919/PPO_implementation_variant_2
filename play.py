from GLOBALS import *
from alg_env_wrapper import SingleAgentEnv


def play(env, times: int = 1, model: nn.Module = None):
    state = env.reset()
    game = 0
    total_reward = 0
    while game < times:
        if model:
            action = model(state)
        else:
            action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        total_reward += reward.item()
        env.render()
        if done:
            state = env.reset()
            game += 1
            print(f'finished game {game} with a total reward: {total_reward}')
            total_reward = 0
        else:
            state = next_state
    env.close()


if __name__ == '__main__':
    # torch.save(actor, f'{SAVE_PATH}/actor.pt')
    # torch.save(target_actor, f'{SAVE_PATH}/target_actor.pt')
    actor_model = torch.load(f'{SAVE_PATH}/actor.pt')
    actor_model.eval()
    curr_env = SingleAgentEnv(env_name=ENV_NAME)
    play(curr_env, 10, actor_model)
