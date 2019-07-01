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
                xlabel='Timesteps',
                ylabel='Episodic Reward',
                fillarea=False,
                markers=True,
                markersize=4,
                dash=np.array(['dot', 'dot', 'dot','dot']),
                opacity=.25
            )
        )
    else:
        viz.line(
            X=np.array([ep]),
            Y=np.array([ep_reward]),
            win=win,
            # name='all',
            update='append'
        )
        viz.line(
            X=np.array([ep]),
            Y=np.array([ep_reward.mean()]),
            win=win,
            name='mean',
            update='append',
            opts=dict(
                fillarea=False,
                markers=True,
                markersize=7
            )
        )
