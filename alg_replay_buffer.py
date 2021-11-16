from GLOBALS import *


class ReplayBuffer:
    def __init__(self):
        self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    def __len__(self):
        return len(self.replay_buffer)

    def __getitem__(self, index):
        item = self.replay_buffer[index]
        # state, action, reward, done, new_state = item
        # return state, action, reward, done, new_state
        # return item.state, item.action, item.reward, item.done, item.new_state
        return item

    def append(self, item):
        self.replay_buffer.append(item)

    def sample(self, n=1):
        # random sample
        return random.sample(self.replay_buffer, n)

