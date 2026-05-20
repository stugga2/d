import pybullet as p
import pybullet_data
import math
import time
import matplotlib.pyplot as plt
import numpy as np

# 1. Initialize simulation
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setRealTimeSimulation(0)
p.loadURDF("plane.urdf")

# 2. Robot parameters
base_mass = 1.0
wheel_mass = 0.1
wheel_radius = 0.04
wheel_length = 0.015
mount_radius = 0.12
# 3 wheels at 120° intervals
angles = [0, 2*math.pi/3, 4*math.pi/3]

# 3. Create collision shapes
base_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.03])
wheel_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=wheel_radius, height=wheel_length)

# 4. Rotate each wheel cylinder so its local Z/cylinder axis is radial.
# The revolute joint axis is expressed in the wheel's local frame, so [0, 0, 1]
# makes every wheel spin around its own cylinder axle.
linkOrientations = [p.getQuaternionFromEuler([0, math.pi/2, a]) for a in angles]
linkJointAxis = [[0, 0, 1]] * 3

# 5. Build robot
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

# 6. Set friction properties (omni-wheel approximation)
for i in range(3):
    p.changeDynamics(robot_id, i,
                     lateralFriction=1.5,
                     rollingFriction=0.0,
                     spinningFriction=0.0,
                     anisotropicFriction=[1.0, 1.0, 0.02],
                     frictionAnchor=True)
p.changeDynamics(robot_id, -1, lateralFriction=0.05, rollingFriction=0.01)

# 7. Inverse kinematics for 3-omni-wheel (Kiwi drive) platform
def get_wheel_velocities(vx, vy, wz):
    wheel_vels = []
    for theta in angles:
        # Standard holonomic kinematics with tangential wheel drive directions.
        v_i = (-math.sin(theta)*vx + math.cos(theta)*vy + mount_radius*wz) / wheel_radius
        wheel_vels.append(v_i)
    return wheel_vels

def world_to_robot_velocity(vx_world, vy_world, yaw):
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return (
        cos_yaw * vx_world + sin_yaw * vy_world,
        -sin_yaw * vx_world + cos_yaw * vy_world
    )

# 8. Simulation loop
print("Starting simulation...")
print("Test: Straight world-forward translation while rotating")

###
target_pos = [-4.0, -4.0] # задаём координаты
kp_linear = 0.8     # коэффициент для скорости вперед/назад
kp_yaw = 1.8        # коэффициент для поворота к цели
max_speed = 0.3     # ограничение линейной скорости
max_wz = 1.5        # ограничение угловой скорости
stop_threshold = 0.05   # точность остановки
###

# Массивы для записи истории
time_hist = []
pos_hist = []      # [x, y, yaw]
desired_hist = []  # целевые точки (для отображения)
step_time = 0.0
dt_sim = 1./240.

# Целевая траектория для отображения (прямая линия к target_pos)
start_pos = [0.0, 0.0, 0.0]  # начальная позиция робота
target_point = [target_pos[0], target_pos[1], 0.0]  # конечная цель

for step in range(3000):
    ###
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    x, y = pos[0], pos[1]
    _, _, yaw = p.getEulerFromQuaternion(orn)
    
    dx = x - target_pos[0]
    dy = y - target_pos[1]
    distance = math.sqrt(dx*dx + dy*dy)

    # Запись данных для графиков
    time_hist.append(step_time)
    pos_hist.append([x, y, yaw])
    
    # Вычисление желаемой позиции (линейная интерполяция от старта к цели)
    t_norm = min(1.0, distance / max(0.01, math.hypot(start_pos[0]-target_pos[0], start_pos[1]-target_pos[1])))
    desired_hist.append([
        start_pos[0] + t_norm * (target_pos[0] - start_pos[0]),
        start_pos[1] + t_norm * (target_pos[1] - start_pos[1]),
        0.0
    ])
    
    if distance < stop_threshold:
        vx_world, vy_world, wz = 0.0, 0.0, 0.0
        for i in range(3):
            p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, targetVelocity=0.0, force=20.0)
        p.stepSimulation()
        print(f"({target_pos[0]}, {target_pos[1]})")
        break
    else:
        angle_to_target = math.atan2(dy, dx)
        yaw_error = angle_to_target - yaw
        
        yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))
        
        vx_world = max(-max_speed, min(max_speed, kp_linear * dx))
        vy_world = max(-max_speed, min(max_speed, kp_linear * dy))
        wz = max(-max_wz, min(max_wz, kp_yaw * yaw_error))
    ###
    _, orn = p.getBasePositionAndOrientation(robot_id)
    _, _, yaw = p.getEulerFromQuaternion(orn)
    vx_body, vy_body = world_to_robot_velocity(vx_world, vy_world, yaw)

    target_vels = get_wheel_velocities(vx_body, vy_body, wz)
    for i in range(3):
        p.setJointMotorControl2(
            robot_id, i,
            p.VELOCITY_CONTROL,
            targetVelocity=target_vels[i],
            force=20.0
        )

    p.stepSimulation()
    step_time += dt_sim
    time.sleep(1./240.)

    if step % 250 == 0:
        pos, orn = p.getBasePositionAndOrientation(robot_id)
        # Extract yaw for rotation monitoring
        _, _, yaw = p.getEulerFromQuaternion(orn)
        print(f"Step {step:4d} | Pos: ({pos[0]:.2f}, {pos[1]:.2f}) | Yaw: {math.degrees(yaw):.1f}°")

# Преобразование в numpy массивы
time_hist = np.array(time_hist)
pos_hist = np.array(pos_hist)
desired_hist = np.array(desired_hist)

# Рисунок 1: XY траектория
plt.figure("XY траектория", figsize=(10, 8))
plt.plot(pos_hist[:,0], pos_hist[:,1], 'b-', linewidth=2, label='Реальная траектория')
plt.plot(desired_hist[:,0], desired_hist[:,1], 'r--', linewidth=1.5, label='Желаемая траектория')
plt.plot([target_pos[0]], [target_pos[1]], 'go', markersize=10, label='Целевая точка')
plt.plot([0], [0], 'ro', markersize=8, label='Старт')
plt.xlabel('X (м)', fontsize=12)
plt.ylabel('Y (м)', fontsize=12)
plt.title('Траектория движения робота в плоскости XY', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()

# Рисунок 2: Положение во времени
plt.figure("Положение", figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(time_hist, pos_hist[:,0], 'b-', linewidth=1.5, label='x факт')
plt.plot(time_hist, -desired_hist[:,0]-4, 'r--', linewidth=1.5, label='x желаемое')
plt.ylabel('X (м)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.title('Координата X(t)', fontsize=12)

plt.subplot(2, 1, 2)
plt.plot(time_hist, pos_hist[:,1], 'b-', linewidth=1.5, label='y факт')
plt.plot(time_hist, -desired_hist[:,1]-4, 'r--', linewidth=1.5, label='y желаемое')
plt.xlabel('Время (с)', fontsize=12)
plt.ylabel('Y (м)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.title('Координата Y(t)', fontsize=12)

plt.tight_layout()

# Рисунок 3: Ошибка позиции
plt.figure("Ошибка позиции", figsize=(10, 6))
error_dist = np.sqrt((pos_hist[:,0] - target_pos[0])**2 + (pos_hist[:,1] - target_pos[1])**2)
plt.plot(time_hist, error_dist, 'g-', linewidth=1.5)
plt.xlabel('Время (с)', fontsize=12)
plt.ylabel('Расстояние до цели (м)', fontsize=12)
plt.title('Ошибка позиции во времени', fontsize=14)
plt.grid(True, alpha=0.3)
plt.axhline(y=stop_threshold, color='r', linestyle='--', label=f'Порог остановки ({stop_threshold} м)')
plt.legend(fontsize=10)
plt.tight_layout()

plt.show()
