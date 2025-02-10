# Code adapted from https://prompt-to-prompt.github.io/
import abc


class EmptyControl:
    def step_callback(self, x_t):
        return x_t
    def between_steps(self):
        return
    def __call__(self, attn, layer_n: str):
        return attn


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    def between_steps_inject(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0 #self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, layer_n: str):
        raise NotImplementedError

    def __call__(self, attn, layer_n: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            attn = self.forward(attn, layer_n)
        return attn

    def check_next_step(self):
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            if self.is_inject:
                self.between_steps_inject()
            else:
                self.between_steps()

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.is_inject = False


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {}

    def forward(self, attn, layer_n: str):
        if attn is None:
            attn = self.attn_store[self.cur_step][layer_n]
        else:
            self.step_store[layer_n] = attn.cpu()
        return attn

    def between_steps(self):
        self.attn_store[self.cur_step - 1] = self.step_store
        self.step_store = self.get_empty_store()

    def between_steps_inject(self):
        self.step_store = self.get_empty_store()

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attn_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attn_store = {}
