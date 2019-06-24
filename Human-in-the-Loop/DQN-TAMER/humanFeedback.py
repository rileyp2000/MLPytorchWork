import sys, gym, time, math


class Human():
    
    def evaluate(s):
        x, y, vx, vy, ang, angv, _, _ = s
        human_feedback = 0
        if math.sqrt(math.pow(x,2) + math.pow(y,2)) < .25:
            human_feedback += 10
        if ang < math.radians(5) and angv < .012:
            human_feedback += 5
        if ang > math.radians(45):
            human_feedback -= 20

        return human_feedback
