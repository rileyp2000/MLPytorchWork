import numpy as np
from visdom import Visdom

viz = Visdom()

win = None

def update_viz(ep, ep_reward, algo):
    global win

    if win is None:
        win = viz.line(
            X=np.array([ep]),
            Y=np.array([ep_reward]),
            win=algo,
            opts=dict(
                title=algo,
                fillarea=False
            )
        )
    else:
        viz.line(
            X=np.array([ep]),
            Y=np.array([ep_reward]),
            win=win,
            update='append',
            xaxis='Episodes',
            yaxis='Reward'
        )
