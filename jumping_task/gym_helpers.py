import gym

def create_gym_environment(environment_name=None, version='v0', **kwargs):
    assert environment_name is not None
    full_game_name = f'{environment_name}-{version}'
    env = gym.make(full_game_name, **kwargs)
    return env
