from networks.modeling_llada import LLaDAModelLM

class LLaDAARCOT(LLaDAModelLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)