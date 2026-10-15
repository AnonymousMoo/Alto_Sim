import time
import mujoco
import mujoco.viewer

# Custom target angles (radians) for the two servo joints
SERVO_L2_L_ANGLE = -0.72
SERVO_L2_R_ANGLE = 0.0

m = mujoco.MjModel.from_xml_path("scene.xml")
d = mujoco.MjData(m)

d.actuator("servo_l2_L").ctrl[0] = SERVO_L2_L_ANGLE
d.actuator("servo_l2_R").ctrl[0] = SERVO_L2_R_ANGLE

print_every = 50  # print once every 50 physics steps (~10 Hz at timestep=0.002)
step_count = 0

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        step_start = time.time()

        mujoco.mj_step(m, d)
        step_count += 1

        if step_count % print_every == 0:
            print(
                "acc:", d.sensor("imu_acc").data,
                "gyro:", d.sensor("imu_gyro").data,
            )

        viewer.sync()

        remaining = m.opt.timestep - (time.time() - step_start)
        if remaining > 0:
            time.sleep(remaining)
