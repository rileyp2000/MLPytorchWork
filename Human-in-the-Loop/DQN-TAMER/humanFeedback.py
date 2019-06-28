import sys, gym, time, math, random

class Human():
    def __init__(self, r):
        random.seed(r)

    def evaluate(self, s):
        rand = random.random()
        x, y, vx, vy, ang, angv, _, _ = s
        human_feedback = 0
        if rand < .05:
            if math.sqrt(math.pow(x,2) + math.pow(y,2)) < .2:
                human_feedback += 25
                print('dist')
                if ang < math.radians(2) and angv < .005:
                    human_feedback += 25
                    print('ang')
        if rand < .1:
            if ang > math.radians(45):
                human_feedback -= 25
                print('ang bad')

        return human_feedback
