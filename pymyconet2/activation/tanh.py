from ..activation.single_stage import SingleStage

class TanH(SingleStage):
    def __init__(self):
        super().__init__("tanh")