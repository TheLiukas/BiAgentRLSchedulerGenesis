import numpy as np
import genesis as gs

class Obstacle_Gen:
    '''
        Class to generate box obstacles in genesis scene
    '''
    def __init__(self,r_seed, n_obstacles,scene:gs.Scene):
        self.r_seed=r_seed
        self.n_obstacles=n_obstacles
        self.scene=scene
        self.ranGen=np.random.RandomState(self.r_seed)
    
    def generateObstacles(self):
        obstacles=list()
        for i in range(self.n_obstacles):
            pos_arr:np.array=self.ranGen.random(3)
            pos=tuple(pos_arr)
            print(pos)
            obstacle :gs.morphs.Box= self.scene.add_entity(gs.morphs.Box(pos=pos,fixed=True,size=(0.1,0.1,0.1)))
            obstacles.append(obstacle)
        return obstacles