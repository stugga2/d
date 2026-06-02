import pybullet as p
import pybullet_data
import math
import time
import numpy as np
import matplotlib.pyplot as plt

# симуляция
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setRealTimeSimulation(0)
p.loadURDF("plane.urdf")

# параметры модели омниробота
base_mass = 1.0
wheel_mass = 0.1
wheel_radius = 0.04
wheel_length = 0.015
mount_radius = 0.12

# углы для расположения омниколёс
angles = [0, 2*math.pi/3, 4*math.pi/3]

# геометрические модели с коллизиями
base_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.03])
wheel_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=wheel_radius, height=wheel_length)

# поворачиваем каждое колесо на нужный угол
linkOrientations = [p.getQuaternionFromEuler([0, math.pi/2, a]) for a in angles]
linkJointAxis = [[0, 0, 1]] * 3

# построение модели омниробота
robot_id = p.createMultiBody(
    baseMass=base_mass,
    baseCollisionShapeIndex=base_shape,
    baseVisualShapeIndex=-1,
    basePosition=[0, 0, wheel_radius],
    baseOrientation=[0, 0, 0, 1],
    linkMasses=[wheel_mass]*3,
    linkCollisionShapeIndices=[wheel_shape]*3,
    linkVisualShapeIndices=[-1]*3,
    linkPositions=[[mount_radius*math.cos(a), mount_radius*math.sin(a), 0] for a in angles],
    linkOrientations=linkOrientations,
    linkInertialFramePositions=[[0, 0, 0]]*3,
    linkInertialFrameOrientations=[[0, 0, 0, 1]]*3,
    linkParentIndices=[0, 0, 0],
    linkJointTypes=[p.JOINT_REVOLUTE]*3,
    linkJointAxis=linkJointAxis
)

# настройка трения для симуляции омниколёс
for i in range(3):
    p.changeDynamics(robot_id, i,
                     lateralFriction=1.5,
                     rollingFriction=0.0,
                     spinningFriction=0.0,
                     anisotropicFriction=[1.0, 1.0, 0.02],
                     frictionAnchor=True)
p.changeDynamics(robot_id, -1, lateralFriction=0.05, rollingFriction=0.01)

# обратная кинематика для модели трёхколёсного омниробота
def get_wheel_velocities(vx_body, vy_body, wz):
    wheel_vels = []
    for theta in angles:
        v_i = (-math.sin(theta)*vx_body + math.cos(theta)*vy_body + mount_radius*wz) / wheel_radius
        wheel_vels.append(v_i)
    return [-v for v in wheel_vels]

def world_to_robot_velocity(vx_world, vy_world, yaw):
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return (cos_yaw * vx_world + sin_yaw * vy_world,
            -sin_yaw * vx_world + cos_yaw * vy_world)

# полиномиальная траектория

waypoints = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 2.0, math.radians(90)],
    [-1.0, 3.0, math.radians(-60)],
    [4.0, 0.0, math.radians(-180)],
    [-2.0, -2.0, math.radians(0)]
])

def unwrap_yaw(wp):
    yaw = wp[:, 2].copy()
    for i in range(1, len(yaw)):
        diff = yaw[i] - yaw[i-1]
        if diff > math.pi:
            yaw[i] -= 2*math.pi
        elif diff < -math.pi:
            yaw[i] += 2*math.pi
    wp[:, 2] = yaw
    return wp
waypoints = unwrap_yaw(waypoints)

velocities = np.zeros_like(waypoints)
T_segment = 5.0

coeffs = []
num_segments = len(waypoints) - 1
for k in range(num_segments):
    p0 = waypoints[k]
    p1 = waypoints[k+1]
    v0 = velocities[k]
    v1 = velocities[k+1]
    T = T_segment
    a = (2*p0 + (v0+v1)*T - 2*p1) / T**3
    b = (3*p1 - 3*p0 - 2*v0*T - v1*T) / T**2
    c = v0
    d = p0
    seg_coeffs = np.vstack([a, b, c, d]).T
    coeffs.append(seg_coeffs)

# управление

dt = 1/240.0
sim_time = 0.0
segment_idx = 0
segment_start_time = 0.0

# настройка параметров

kp_linear = 10.5
kp_yaw = 1.8
max_speed = 1.9
max_wz = 2.5

p.resetBasePositionAndOrientation(robot_id,
                                  [waypoints[0,0], waypoints[0,1], wheel_radius],
                                  p.getQuaternionFromEuler([0, 0, waypoints[0,2]]))

for i, wp in enumerate(waypoints):
    print(f"  {i}: ({wp[0]:.2f}, {wp[1]:.2f}) угол {math.degrees(wp[2]):.1f}°")

time_hist, pos_hist, desired_hist, error_hist = [], [], [], []
last_print = 0.0
    
while segment_idx < num_segments:
    t_seg = sim_time - segment_start_time

    if t_seg >= T_segment:
        segment_idx += 1
        segment_start_time = sim_time
        continue

    coeff = coeffs[segment_idx]
    t = t_seg
    pos_des = coeff[:,3] + coeff[:,2]*t + coeff[:,1]*t**2 + coeff[:,0]*t**3

    pos_act_raw, orn = p.getBasePositionAndOrientation(robot_id)
    x_act, y_act = pos_act_raw[0], pos_act_raw[1]
    _, _, yaw_act = p.getEulerFromQuaternion(orn)

    dx = pos_des[0] - x_act
    dy = pos_des[1] - y_act
    dyaw = pos_des[2] - yaw_act
    dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))

    vx_world = kp_linear * dx
    vy_world = kp_linear * dy
    wz = kp_yaw * dyaw

    if segment_idx == 0:
        wz = 0.4
        
    v_lin = math.hypot(vx_world, vy_world)
    if v_lin > max_speed:
        vx_world = vx_world / v_lin * max_speed
        vy_world = vy_world / v_lin * max_speed
    wz = max(-max_wz, min(max_wz, wz))

    vx_body, vy_body = world_to_robot_velocity(vx_world, vy_world, yaw_act)
    wheel_targets = get_wheel_velocities(vx_body, vy_body, wz)
    
    for i in range(3):
        p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL,
                                targetVelocity=wheel_targets[i], force=50.0)

    p.stepSimulation()
    time.sleep(dt)

    time_hist.append(sim_time)
    pos_hist.append([x_act, y_act, yaw_act])
    desired_hist.append(pos_des)
    error_dist = math.hypot(dx, dy)
    error_hist.append(error_dist)

    if sim_time - last_print >= 0.5:
        print(f"t={sim_time:.2f}s | факт: ({x_act:.2f}, {y_act:.2f}) "
              f"угол={math.degrees(yaw_act):.1f}° | ошибка={error_dist:.3f} м | "
              f"цель: ({pos_des[0]:.2f}, {pos_des[1]:.2f}) угол={math.degrees(pos_des[2]):.1f}°")
        last_print = sim_time

    sim_time += dt

for i in range(3):
    p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, targetVelocity=0.0, force=20.0)
time.sleep(1)

# графики
time_hist = np.array(time_hist)
pos_hist = np.array(pos_hist)
desired_hist = np.array(desired_hist)

plt.figure("XY траектория")
plt.plot(pos_hist[:,0], pos_hist[:,1], 'b-', label='Реальная')
plt.plot(desired_hist[:,0], desired_hist[:,1], 'r--', label='Желаемая')
plt.plot(waypoints[:,0], waypoints[:,1], 'go', markersize=8, label='Опорные точки')
plt.xlabel('X (м)')
plt.ylabel('Y (м)')
plt.legend()
plt.grid(True)
plt.axis('equal')

plt.figure("Изменение положения и ориентации")
plt.subplot(3,1,1)
plt.plot(time_hist, pos_hist[:,0], label='x факт')
plt.plot(time_hist, desired_hist[:,0], 'r--', label='x желаемое')
plt.ylabel('X (м)')
plt.legend()
plt.grid(True)
plt.subplot(3,1,2)
plt.plot(time_hist, pos_hist[:,1], label='y факт')
plt.plot(time_hist, desired_hist[:,1], 'r--', label='y желаемое')
plt.ylabel('Y (м)')
plt.legend()
plt.grid(True)
plt.subplot(3,1,3)
plt.plot(time_hist, np.degrees(pos_hist[:,2]), label='угол факт')
plt.plot(time_hist, np.degrees(desired_hist[:,2]), 'r--', label='угол желаемый')
plt.xlabel('Время (с)')
plt.ylabel('Угол (град)')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.figure("Изменение ошибки")
plt.plot(time_hist, error_hist)
plt.xlabel('Время (с)')
plt.ylabel('Ошибка позиции (м)')
plt.grid(True)

plt.show()
p.disconnect()
