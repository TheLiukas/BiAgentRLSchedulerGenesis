import argparse
import numpy as np
import genesis as gs
import time
import scripts.utils.gen_obstacles as gen_obstacles


class DroneController:
    def __init__(self):
        self.thrust = 14468.429183500699  # Base hover RPM - constant hover
        self.rotation_delta = 200  # Differential RPM for rotation
        self.thrust_delta = 10  # Amount to change thrust by when accelerating/decelerating
        self.running = True
        self.rpms = [self.thrust] * 4
        self.pressed_keys = set()


    def update_thrust(self):
        # Store previous RPMs for debugging
        prev_rpms = self.rpms.copy()

        # Reset RPMs to hover thrust
        self.rpms = [self.thrust] * 4

        

        self.rpms = np.clip(self.rpms, 0, 25000)

        # Debug print if any RPMs changed
        if not np.array_equal(prev_rpms, self.rpms):
            print(f"RPMs changed from {prev_rpms} to {self.rpms}")

        return self.rpms


def run_sim(scene, drone, controller):
    while controller.running:
        try:
            # Update drone with current RPMs
            rpms = controller.update_thrust()
            drone.set_propellels_rpm(rpms)

            # Update physics
            scene.step()

            time.sleep(1 / 60)  # Limit simulation rate
        except Exception as e:
            print(f"Error in simulation loop: {e}")

    if scene.viewer:
        scene.viewer.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=True, help="Enable visualization (default: True)")
    args = parser.parse_args()

    # Initialize Genesis
    gs.init(backend=gs.gpu)

    # Create scene with initial camera view
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(0.0, -4.0, 2.0),  # Now behind the drone (negative Y)
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=45,
        max_FPS=60,
    )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0, 0, -9.81),
        ),
        viewer_options=viewer_options,
        show_viewer=args.vis,
    )

    # Add entities
    plane = scene.add_entity(gs.morphs.Plane())
    obstacle_gen=gen_obstacles.Obstacle_Gen(42,2,scene)
    obstacle=obstacle_gen.generateObstacles()
    #obstacle.
    drone = scene.add_entity(
        morph=gs.morphs.Drone(
            file="urdf/drones/cf2x.urdf",
            pos=(0.0, 0, 0.5),  # Start a bit higher
        ),
    )

    #scene.viewer.follow_entity(drone)

    # Build scene
    scene.build()

    # Initialize controller
    controller = DroneController()

    # Print control instructions



    run_sim(scene, drone, controller)


if __name__ == "__main__":
    main()
