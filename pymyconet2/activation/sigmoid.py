from ..activation.single_stage import SingleStage

class Sigmoid(SingleStage):
    def __init__(self):
        super().__init__("sigmoid")