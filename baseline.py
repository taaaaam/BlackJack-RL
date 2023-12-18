from classes import Agent

class PlayerBASE(Agent):
    def __init__(self, epsilon = 0.1, discount_rate = 1):
        super().__init__()

    def play(self, epoch = -1, training_flag = True):
        while self.hand.value <= 14:
            self.deal_card()
        return self.hand.value
