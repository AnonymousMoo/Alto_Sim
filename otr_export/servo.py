import time
import numpy as np
import mujoco
import mujoco.viewer

# ---- Edit these ----
# (min_angle, max_angle) in radians for each servo
SERVO_L2_L_RANGE = (-0.72, 0.0)
SERVO_L2_R_RANGE = (0.0, 0.72)

SWEEP_PERIOD = 4.0    # seconds for one full min -> max -> min cycle
PHASE_OFFSET = 0.0    # radians; shifts R relative to L (np.pi = opposite motion)
PRINT_EVERY = 50      # physics steps between prints (0 to disable)
# --------------------

m = mujoco.MjModel.from_xml_path("scene.xml")
d = mujoco.MjData(m)

act_L = m.actuator("servo_l2_L").id
act_R = m.actuator("servo_l2_R").id


def sweep(t, lo, hi, phase=0.0):
    """Smooth sinusoidal sweep between lo and hi."""
    s = 0.5 * (1 - np.cos(2 * np.pi * t / SWEEP_PERIOD + phase))  # 0 -> 1 -> 0
    return lo + s * (hi - lo)


def clamp_to_ctrlrange(act_id, value):
    if m.actuator_ctrllimited[act_id]:
        lo, hi = m.actuator_ctrlrange[act_id]
        return float(np.clip(value, lo, hi))
    return value


step_count = 0

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        step_start = time.time()

        t = d.time
        d.ctrl[act_L] = clamp_to_ctrlrange(act_L, sweep(t, *SERVO_L2_L_RANGE))
        d.ctrl[act_R] = clamp_to_ctrlrange(act_R, sweep(t, *SERVO_L2_R_RANGE, PHASE_OFFSET))

        mujoco.mj_step(m, d)
        step_count += 1

        if PRINT_EVERY and step_count % PRINT_EVERY == 0:
            print(
                f"t={d.time:.2f}",
                "ctrl L/R:", round(d.ctrl[act_L], 3), round(d.ctrl[act_R], 3),
                "acc:", d.sensor("imu_acc").data,
                "gyro:", d.sensor("imu_gyro").data,
            )

        viewer.sync()

        remaining = m.opt.timestep - (time.time() - step_start)
        if remaining > 0:
            time.sleep(remaining)
            
