"""
Original version available on : https://github.com/AliFakhry/DQN


Easiest continuous control task to learn from pixels, a top-down racing
environment.
Discrete control is reasonable in this environment as well, on/off
discretization is fine.
State consists of STATE_W x STATE_H pixels.
The reward is -0.1 every frame and +1000/N for every track tile visited, where
N is the total number of tiles visited in the track. For example, if you have
finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points.
The game is solved when the agent consistently gets 900+ points. The generated
track is random every episode.
The episode finishes when all the tiles are visited. The car also can go
outside of the PLAYFIELD -  that is far off the track, then it will get -100
and die.
Some indicators are shown at the bottom of the window along with the state RGB
buffer. From left to right: the true speed, four ABS sensors, the steering
wheel position and gyroscope.
To play yourself (it's rather fast for humans), type:
python gym/envs/box2d/car_racing.py
Remember it's a powerful rear-wheel drive car -  don't press the accelerator
and turn at the same time.
Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.
"""
import math
import numpy as np

import Box2D
from Box2D.b2 import fixtureDef
from Box2D.b2 import polygonShape
from Box2D.b2 import contactListener

import gym
from gym import spaces
from car_dynamics import Car, SENSOR_NB
from gym.utils import seeding, EzPickle

import pyglet
from pyglet import gl

from gym import error, spaces, utils
from gym.utils import seeding


STATE_W = 96   # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0             # Track scale
TRACK_RAD = 900/SCALE   # Track is heavily morphed circle with this radius
PLAYFIELD = 2000/SCALE  # Game over boundary
FPS = 50               # Frames per second
ZOOM = 4                # Camera zoom
ZOOM_FOLLOW = False     # Set to False for fixed view (don't use zoom) 

TRACK_DETAIL_STEP = 21/SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 60/SCALE
BORDER = 0/SCALE
BORDER_MIN_COUNT = 10   
Amount_Left = 0

OBSTACLE_PROB = 1/25  # modifiable à l'init de CarRacing

ROAD_COLOR = [0.4, 0.4, 0.4]

class FrictionDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        # Change la couleur des tiles parcourues
        
        if u1 in self.env.car.wheels or u2 in self.env.car.wheels:
            tile.color[0] = ROAD_COLOR[0]
            tile.color[1] = ROAD_COLOR[1]
            tile.color[2] = ROAD_COLOR[2]

        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            obj.tiles.add(tile)
            if not tile.road_visited and (u1 in self.env.car.wheels or u2 in self.env.car.wheels):
                tile.road_visited = True

                self.env.tile_visited_count += 1
                global Amount_Left
                Amount_Left= (len(self.env.track)- self.env.tile_visited_count)
        else:
            obj.tiles.remove(tile)


class CarRacing(gym.Env, EzPickle):
    
    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels'],
        'video.frames_per_second': FPS
    }

    def __init__(self, obstacle_prob, verbose=1, sensors_activated = True):
        EzPickle.__init__(self)
        self.obstacles_positions = [] # sera rempli de 4-tuples contenant la position de chaque mur
        self.seed()
        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World(
            (0, 0),
            contactListener=self.contactListener_keepref
        )
        self.viewer = None
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.times_succeeded=0
        self.verbose = verbose
        self.sensors_activated = sensors_activated
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )

        self.action_space = spaces.Box(np.array([-1, 0, 0]),
                                       np.array([+1, +1, +1]),
                                       dtype=np.float32)  # steer, gas, brake

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(STATE_H, STATE_W, 3),
            dtype=np.uint8
        )
        self.random_obs = 0
        global OBSTACLE_PROB 
        OBSTACLE_PROB = obstacle_prob

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        self.car.destroy()

    def _create_track(self):

        CHECKPOINTS = 12 # 12 = nombre de virages (11) + le départ (1)

        # Create checkpoints
        checkpoints = []
        self.obstacles_positions = []
        for c in range(CHECKPOINTS):
            noise = self.np_random.uniform(0, 2 * 3.14159 * 1 / CHECKPOINTS)
            alpha = 2 * 3.14159 * c / CHECKPOINTS + noise
            rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)

            if c == 0:
                alpha = 0
                rad = 1.5 * TRACK_RAD
            if c == CHECKPOINTS - 1:
                alpha = 2 * 3.14159 * c / CHECKPOINTS
                self.start_alpha = 2 * 3.14159 * (-0.5) / CHECKPOINTS
                rad = 1.5 * TRACK_RAD

            checkpoints.append(
                (alpha, rad * math.cos(alpha), rad * math.sin(alpha))
            )
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5 * TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * 3.14159

            while True:  # Find destination from checkpoints
                failed = True

                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break

                if not failed:
                    break

                alpha -= 2*3.14159
                continue

            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            # destination vector projected on rad:
            proj = r1x * dest_dx + r1y * dest_dy
            while beta - alpha > 1.5 * 3.14159:
                beta -= 2 * 3.14159
            while beta - alpha < -1.5 * 3.14159:
                beta += 2 * 3.14159
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3:
                beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3:
                beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False  # Failed
            pass_through_start = track[i][0] > self.start_alpha and track[i-1][0] <= self.start_alpha
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        if self.verbose == 1:
            print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2-i1))
        assert i1 != -1
        assert i2 != -1

        track = track[i1:i2-1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2])) +
            np.square(first_perp_y * (track[0][3] - track[-1][3])))
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # Red-white border on hard turns
        border = [False] * len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i-neg-0][1]
                beta2 = track[i-neg-1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i-neg] |= border[i]

        # Create tiles and obstacles 
        last_obstacle = 15 # pour que le début de course se passe sans obstacle
        for i in range(len(track)):

            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i-1]
            road1_l = (x1 - TRACK_WIDTH*math.cos(beta1), y1 - TRACK_WIDTH*math.sin(beta1))
            road1_r = (x1 + TRACK_WIDTH*math.cos(beta1), y1 + TRACK_WIDTH*math.sin(beta1))
            road2_l = (x2 - TRACK_WIDTH*math.cos(beta2), y2 - TRACK_WIDTH*math.sin(beta2))
            road2_r = (x2 + TRACK_WIDTH*math.cos(beta2), y2 + TRACK_WIDTH*math.sin(beta2))
            vertices = [road1_l, road1_r, road2_r, road2_l]
            
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01*(i%3)
            t.color = [0, 0.128, 0.624, 0.019]
            t.road_visited = False
            t.road_friction = 1.0
            t.fixtures[0].sensor = True
            self.road_poly.append(
                ([road1_l, road1_r, road2_r, road2_l], t.color)
            )
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (x1 + side * TRACK_WIDTH * math.cos(beta1),
                        y1 + side * TRACK_WIDTH * math.sin(beta1))
                b1_r = (x1 + side * (TRACK_WIDTH+BORDER) * math.cos(beta1),
                        y1 + side * (TRACK_WIDTH+BORDER)*math.sin(beta1))
                b2_l = (x2 + side * TRACK_WIDTH * math.cos(beta2),
                        y2 + side * TRACK_WIDTH * math.sin(beta2))
                b2_r = (x2 + side * (TRACK_WIDTH+BORDER) * math.cos(beta2),
                        y2 + side * (TRACK_WIDTH+BORDER) * math.sin(beta2))
                self.road_poly.append(
                    (
                        [b1_l, b1_r, b2_r, b2_l],
                        (1, 1, 1) if i % 2 == 0 else (1, 0, 0)
                    )
                )

            self.random_obs += 1
            self.seed(self.random_obs)
            if (self.np_random.uniform(0, 1) < OBSTACLE_PROB) and (last_obstacle <= 0): # i > 15 pour que la course soit toujours faisable
                last_obstacle = 8 

                deriv_left = self.np_random.uniform(TRACK_WIDTH)
                deriv_right = TRACK_WIDTH - deriv_left
                
                obs1_l = (x1 - (TRACK_WIDTH-deriv_left)*math.cos(beta1), y1 - (TRACK_WIDTH-deriv_left)*math.sin(beta1))
                obs1_r = (x1 + (TRACK_WIDTH-deriv_right)*math.cos(beta1), y1 + (TRACK_WIDTH-deriv_right)*math.sin(beta1))
                obs2_l = (x2 - (TRACK_WIDTH-deriv_left)*math.cos(beta2), y2 - (TRACK_WIDTH-deriv_left)*math.sin(beta2))
                obs2_r = (x2 + (TRACK_WIDTH-deriv_right)*math.cos(beta2), y2 + (TRACK_WIDTH-deriv_right)*math.sin(beta2))
                
                self.obstacles_positions.append((obs1_l, obs1_r, obs2_r, obs2_l))
                vertices = [obs1_l, obs1_r, obs2_r, obs2_l]
                obstacle = fixtureDef(
                    shape=polygonShape(vertices=vertices)
                )
                
                obstacle.userData = obstacle    
                obstacle.color = [0.1568, 0.598, 0.2862, 0]
                
                self.road_poly.append(
                        ([obs1_l, obs1_r, obs2_r, obs2_l], obstacle.color)
                    )
            last_obstacle -= 1

        self.track = track
        return True

    def reset(self):
        self._destroy()
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.road_poly = []

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose == 1:
                print(
                    "retry to generate track (normal if there are not many"
                    "instances of this message)"
                )
        self.car = Car(self.world, *self.track[0][1:4], sensors_activated = self.sensors_activated)

        return self.step([0,0.1,0])[0]

    def step(self, action):

        if action is not None:
            self.car.steer(-action[0])
            self.car.gas(action[1])
            self.car.brake(action[2])
        self.car.step(1.0/FPS)
        self.world.Step(1.0/FPS, 6*30, 2*30)
        self.t += 1.0/FPS

        step_reward = 0
        done = False
        INF = 10000

        state = np.full(SENSOR_NB, INF, dtype=float)

        wall = [False] * len(state)
        if action is not None:  # First step without action, called from reset()
            
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward

            if self.tile_visited_count == len(self.track):
                done = True
                self.times_succeeded+=1
        
            x, y = self.car.hull.position
            
            # Vérification des collisions avec les obstacles:
            for i in range(len(self.obstacles_positions)):
                obs1_l, obs1_r, obs2_r, obs2_l = self.obstacles_positions[i]
                if self.isInsideObstacle((x,y), obs1_l, obs1_r, obs2_l, obs2_r):
                        done = True
                        
            contact = False
            for w in self.car.wheels:
                tiles = w.contacts
                if (tiles.__len__() > 0):
                    LOCATION = "TILE"
                elif (tiles.__len__() == 0):     # vraie détection de sortie de route
                    LOCATION = "GRASS"
                    done = True
                    #step_reward -= 400
                    contact = True

            # SENSORS
            for i in range(len(self.car.sensors)): #check if sensors collide with grass
                tiles = self.car.sensors[i].contacts
                sensor_x = self.car.sensors[i].position.x
                sensor_y = self.car.sensors[i].position.y
                point1 = np.array([sensor_x, sensor_y])
                point2 = np.array([x, y])
                
                self.car.sensors[i].color = (0, 1, 0)
                if not wall[i%SENSOR_NB]:
                    state[i%SENSOR_NB] = INF
                
                if (tiles.__len__() == 0):
                    # Sensor de sortie de circuit                           
                    self.car.sensors[i].color = (0,0,0)

                    if not wall[i%SENSOR_NB]:
                        state[i%SENSOR_NB] = np.linalg.norm(point1-point2)
                        wall[i%SENSOR_NB] = True
                else:
                  
                    # Sensor d'obstacle
                    in_obstacle = False

                    for j in range(len(self.obstacles_positions)):
                        obs1_l, obs1_r, obs2_r, obs2_l = self.obstacles_positions[j]
                        if self.isInsideObstacle((sensor_x,sensor_y), obs1_l, obs1_r, obs2_l, obs2_r):
                            in_obstacle = True
                    if in_obstacle:

                        self.car.sensors[i].color = (0,0,0)
                        if not wall[i%SENSOR_NB]:
                            state[i%SENSOR_NB] = np.linalg.norm(point1-point2)
                            wall[i%SENSOR_NB] = True
                            
                    
        true_speed = np.sqrt(
            np.square(self.car.hull.linearVelocity[0])
            + np.square(self.car.hull.linearVelocity[1])
        )
        return np.append(state, true_speed), step_reward, done


    def isInsideObstacle(self, ref_pos, pos1, pos2, pos3, pos4):
        """
        Vérifie si le point ref_pos se trouve à l'intérieur du quadrilatère composé
        à partir de pos1, pos2, pos3 et pos4
        """
        x, y = ref_pos
        x1, y1 = pos1
        x2, y2 = pos2
        x3, y3 = pos3
        x4, y4 = pos4

        return ((((x2-x1)*(y-y1)) - ((x-x1)*(y2-y1)) <= 0)
            and ((((x1-x3)*(y-y3)) - ((x-x3)*(y1-y3))) <= 0)
            and ((((x3-x4)*(y-y4)) - ((x-x4)*(y3-y4))) <= 0)
            and ((((x4-x2)*(y-y2)) - ((x-x2)*(y4-y2))) <= 0))

    def render(self, mode='human'):
        assert mode in ['human', 'state_pixels', 'rgb_array']
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label(
                '0000',
                font_size=36,
                x=20,
                y=WINDOW_H * 2.5 / 40.00,
                anchor_x='left',
                anchor_y='center',
                color=(255, 255, 255, 255)
            )
            self.tile_label = pyglet.text.Label(
                '000',
                font_size=36,
                x=1,
                y=WINDOW_H * 2 / 2.1,
                anchor_x='left',
                anchor_y='center',
                color=(255, 255, 255, 255)
            )
            self.transform = rendering.Transform()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        if ZOOM_FOLLOW:
            # Animate zoom first second:
            zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        else:
            zoom = ZOOM
        scroll_x = self.car.hull.position[0]
        scroll_y = self.car.hull.position[1]
        angle = -self.car.hull.angle
        vel = self.car.hull.linearVelocity
        if np.linalg.norm(vel) > 0.5:
            angle = math.atan2(vel[0], vel[1])
        self.transform.set_scale(zoom, zoom)
        self.transform.set_translation(
            WINDOW_W/2 - (scroll_x * zoom * math.cos(angle) - scroll_y * zoom * math.sin(angle)),
            WINDOW_H/4 - (scroll_x * zoom * math.sin(angle) + scroll_y * zoom * math.cos(angle))
        )
        self.transform.set_rotation(angle)

        self.car.draw(self.viewer, mode != "state_pixels")

        arr = None
        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()

        win.clear()
        t = self.transform
        if mode == 'rgb_array':
            VP_W = VIDEO_W
            VP_H = VIDEO_H
        elif mode == 'state_pixels':
            VP_W = STATE_W
            VP_H = STATE_H
        else:
            pixel_scale = 1
            if hasattr(win.context, '_nscontext'):
                pixel_scale = win.context._nscontext.view().backingScaleFactor()  # pylint: disable=protected-access
            VP_W = int(pixel_scale * WINDOW_W)
            VP_H = int(pixel_scale * WINDOW_H)

        gl.glViewport(0, 0, VP_W, VP_H)
        t.enable()
        self.render_road()
        for geom in self.viewer.onetime_geoms:
            geom.render()
        self.viewer.onetime_geoms = []
        t.disable()
        self.render_indicators(WINDOW_W, WINDOW_H)

        if mode == 'human':
            win.flip()
            return self.viewer.isopen
        image_data = (
            pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        )
        arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep="")
        arr = arr.reshape(VP_H, VP_W, 4)
        arr = arr[::-1, :, 0:3]

        return arr

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def render_road(self):
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(0, 0, 0, 0)
        k = PLAYFIELD/20.0
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                gl.glVertex3f(k * x + k, k * y + 0, 0)
                gl.glVertex3f(k * x + 0, k * y + 0, 0)
                gl.glVertex3f(k * x + 0, k * y + k, 0)
                gl.glVertex3f(k * x + k, k * y + k, 0)
        for poly, color in self.road_poly:
            gl.glColor4f(color[0], color[1], color[2], 1)
            for p in poly:
                gl.glVertex3f(p[0], p[1], 0)
        gl.glEnd()

    def render_indicators(self, W, H):
        gl.glBegin(gl.GL_QUADS)
        s = W / 40.0
        h = H / 40.0
        gl.glColor4f(0, 0, 0, 1)
        gl.glVertex3f(W, 0, 0)
        gl.glVertex3f(W, 5 * h, 0)
        gl.glVertex3f(0, 5 * h, 0)
        gl.glVertex3f(0, 0, 0)

        def vertical_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place+0) * s, h + h * val, 0)
            gl.glVertex3f((place+1) * s, h + h * val, 0)
            gl.glVertex3f((place+1) * s, h, 0)
            gl.glVertex3f((place+0) * s, h, 0)

        def horiz_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place + 0) * s, 4 * h, 0)
            gl.glVertex3f((place + val) * s, 4 * h, 0)
            gl.glVertex3f((place + val) * s, 2 * h, 0)
            gl.glVertex3f((place + 0) * s, 2 * h, 0)
        true_speed = np.sqrt(
            np.square(self.car.hull.linearVelocity[0])
            + np.square(self.car.hull.linearVelocity[1])
        )
        vertical_ind(5, 0.02*true_speed, (1, 1, 1))
        vertical_ind(7, 0.01*self.car.wheels[0].omega, (0.0, 0, 1))  # ABS sensors
        vertical_ind(8, 0.01*self.car.wheels[1].omega, (0.0, 0, 1))
        vertical_ind(9, 0.01*self.car.wheels[2].omega, (0.2, 0, 1))
        vertical_ind(10, 0.01*self.car.wheels[3].omega, (0.2, 0, 1))
        horiz_ind(20, -10.0 * self.car.wheels[0].joint.angle, (0, 1, 0))
        horiz_ind(30, -0.8 * self.car.hull.angularVelocity, (1, 0, 0))
        gl.glEnd()
        self.score_label.text = "%04i" % self.reward
        self.score_label.draw()

        global Amount_Left

        self.tile_label.text = "%4i" % Amount_Left
        self.tile_label.draw()

    def addSensorBorder(self, x1, y1, x2, y2):
        """
        Fonction qui ajoute les bords de la route comme segments de droite "Segment2D"
        afin d'avoir des points de repère pour nos sensors.
        """
        pt1 = Point2D(x1, y1, evaluate=False)
        pt2 = Point2D(x2, y2, evaluate=False)

        self.sensorBorder.append(Segment2D(pt1,pt2, evaluate=False))

    def setAngleZero(self):
        self.car.hull.angle = 0
    

class CarRacingImage(CarRacing):
    def __init__(self):
        global ZOOM
        super(CarRacingImage, self).__init__()
        self.sensors_activated = False
        ZOOM = 12

    def step(self, action):

        if action is not None:
            self.car.steer(-action[0])
            self.car.gas(action[1])
            self.car.brake(action[2])
        self.car.step(1.0/FPS)
        self.world.Step(1.0/FPS, 6*30, 2*30)
        self.t += 1.0/FPS

        step_reward = 0
        done = False
        INF = 10000

        state = self.render("state_pixels")

        if action is not None:  # First step without action, called from reset()
            
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward

            if self.tile_visited_count == len(self.track):
                done = True
                self.times_succeeded+=1
        
            x, y = self.car.hull.position
            
            # Vérification des collisions avec les obstacles:
            for i in range(len(self.obstacles_positions)):
                obs1_l, obs1_r, obs2_r, obs2_l = self.obstacles_positions[i]
                if self.isInsideObstacle((x,y), obs1_l, obs1_r, obs2_l, obs2_r):
                        done = True
                        
            contact = False
            for w in self.car.wheels:
                tiles = w.contacts
                if (tiles.__len__() > 0):
                    LOCATION = "TILE"
                elif (tiles.__len__() == 0):     # vraie détection de sortie de route
                    LOCATION = "GRASS"
                    done = True
                    contact = True
                    
        return state, step_reward, done

if __name__ == "__main__":
    from pyglet.window import key
    a = np.array([0.0, 0.0, 0.0])

    render = True
    
    if render:
        def key_press(k, mod):
            global restart
            if k == 0xff0d: restart = True
            if k == key.LEFT:  a[0] = -1.0
            if k == key.RIGHT: a[0] = +1.0
            if k == key.UP:    a[1] = +1.0
            if k == key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation

        def key_release(k, mod):
            if k == key.LEFT  and a[0] == -1.0: a[0] = 0
            if k == key.RIGHT and a[0] == +1.0: a[0] = 0
            if k == key.UP:    a[1] = 0
            if k == key.DOWN:  a[2] = 0


    env = CarRacing()
    if render:
        env.render()
        env.viewer.window.on_key_press = key_press
        env.viewer.window.on_key_release = key_release

    isopen = True
    while isopen:
        env.reset()
        #env.setAngleZero()
        print(env.tile_visited_count)

        total_reward = 0.0
        steps = 0
        restart = False

        
        while True:

            s,r, done = env.step(a)

            total_reward += r
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1
            if render:
                isopen = env.render()
            if done or restart or isopen == False:
                break
    env.close()