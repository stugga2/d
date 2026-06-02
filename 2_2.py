import pybullet as p
import pybullet_data
import math
import time
import matplotlib.pyplot as plt
import numpy as np

# симуляция
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81)
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
base_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1,0.1,0.03])
wheel_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=wheel_radius, height=wheel_length)

# поворачиваем каждое колесо на нужный угол
linkOrientations = [p.getQuaternionFromEuler([0, math.pi/2, a]) for a in angles]
linkJointAxis = [[0,0,1]]*3

# построение модели омниробота
robot_id = p.createMultiBody(
    baseMass=base_mass,
    baseCollisionShapeIndex=base_shape,
    baseVisualShapeIndex=-1,
    basePosition=[0,0,wheel_radius],
    baseOrientation=[0,0,0,1],
    linkMasses=[wheel_mass]*3,
    linkCollisionShapeIndices=[wheel_shape]*3,
    linkVisualShapeIndices=[-1]*3,
    linkPositions=[[mount_radius*math.cos(a), mount_radius*math.sin(a), 0] for a in angles],
    linkOrientations=linkOrientations,
    linkInertialFramePositions=[[0,0,0]]*3,
    linkInertialFrameOrientations=[[0,0,0,1]]*3,
    linkParentIndices=[0,0,0],
    linkJointTypes=[p.JOINT_REVOLUTE]*3,
    linkJointAxis=linkJointAxis
)

# настройка трения для симуляции омниколёс
for i in range(3):
    p.changeDynamics(robot_id, i, lateralFriction=1.5, rollingFriction=0.0,
                     spinningFriction=0.0, anisotropicFriction=[1.0,1.0,0.02], frictionAnchor=True)
p.changeDynamics(robot_id, -1, lateralFriction=0.05, rollingFriction=0.01)

def get_wheel_velocities(vx, vy, wz):
    wheel_vels = []
    for theta in angles:
        v_i = (-math.sin(theta)*vx + math.cos(theta)*vy - mount_radius*wz) / wheel_radius
        wheel_vels.append(v_i)
    return wheel_vels

def world_to_robot_velocity(vx_world, vy_world, yaw):
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return (cos_yaw*vx_world + sin_yaw*vy_world,
            -sin_yaw*vx_world + cos_yaw*vy_world)

# задаём параметры
target_pos = [-4.0, -4.0]       # задаём координаты целевой точки
target_yaw = math.radians(60)   # задаём целевую ориентацию
kp_linear = 0.8                 # линейный коэффициент
kp_yaw = 1.5                    # угловой коэффициент
max_speed = 0.3                 # ограничение линейной скорости
max_wz = 2.0                    # ограничение угловой скорости
stop_threshold = 0.05           # точность остановки по расстоянию
angle_threshold = 0.05          # точность остановки по ориентации

time_hist = []
pos_hist = []
desired_hist = []
step_time = 0.0
dt_sim = 1./240.
start_pos = [0.0,0.0,0.0]

target_pos[0], target_pos[1] = -target_pos[0], -target_pos[1]
for step in range(3500):
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    x,y = pos[0], pos[1]
    _,_,yaw = p.getEulerFromQuaternion(orn)
    
    dx = target_pos[0] - x
    dy = target_pos[1] - y
    dist = math.hypot(dx, dy)
    dyaw = target_yaw - yaw
    dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))   # нормализация
    
    time_hist.append(step_time)
    pos_hist.append([x,y,yaw])
    
    total_dist = math.hypot(target_pos[0]-start_pos[0], target_pos[1]-start_pos[1])
    t_norm = min(1.0, dist / max(0.01, total_dist))
    desired_hist.append([start_pos[0] + t_norm*(target_pos[0]-start_pos[0]),
                         start_pos[1] + t_norm*(target_pos[1]-start_pos[1]), 0])
    
    # условие остановки
    if dist < stop_threshold and abs(dyaw) < angle_threshold:
        for i in range(3):
            p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, targetVelocity=0, force=50)
        p.stepSimulation()
        print(f"\n({x:.3f}, {y:.3f}), {math.degrees(yaw):.1f}°")
        break
    
    # линейная скорость в глобальной системе координат
    vx_world = max(-max_speed, min(max_speed, kp_linear * dx))
    vy_world = max(-max_speed, min(max_speed, kp_linear * dy))
    # угловая скорость (пропорционально ошибке)
    wz = max(-max_wz, min(max_wz, kp_yaw * dyaw))
    
    # пересчёт в локальную систему координат
    vx_body, vy_body = world_to_robot_velocity(vx_world, vy_world, yaw)
    
    # скорости колёс
    target_vels = get_wheel_velocities(vx_body, vy_body, wz)
    
    for i in range(3):
        p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, targetVelocity=target_vels[i], force=50)
    
    p.stepSimulation()
    step_time += dt_sim
    time.sleep(1./240.)
    
    if step % 100 == 0:
        print(f"{step:4d} {step_time:6.2f} {x:6.2f} {y:6.2f} {math.degrees(yaw):6.1f}  "
              f"{vx_world:5.2f} {vy_world:5.2f} {vx_body:5.2f} {vy_body:5.2f} {wz:5.2f}   "
              f"[{target_vels[0]:6.1f} {target_vels[1]:6.1f} {target_vels[2]:6.1f}]")

# построение графиков
time_hist = np.array(time_hist)
pos_hist = np.array(pos_hist)
desired_hist = np.array(desired_hist)

plt.figure("XY траектория", figsize=(10,8))
plt.plot(pos_hist[:,0], pos_hist[:,1], 'b-', label='Траектория')
#plt.plot(desired_hist[:,0], desired_hist[:,1], 'r--', label='Желаемая')
plt.plot(-target_pos[0], -target_pos[1], 'go', markersize=10, label='Цель')
plt.plot(0,0,'ro', label='Старт')
plt.xlabel('X (м)'); plt.ylabel('Y (м)')
plt.title('Траектория движения')
plt.legend(); plt.grid(); plt.axis('equal')
plt.tight_layout()

plt.figure("Ошибка ориентации", figsize=(10, 6))
error_yaw = np.abs(np.array([np.arctan2(np.sin(ang), np.cos(ang)) for ang in pos_hist[:,2] - target_yaw]))
plt.plot(time_hist, error_yaw, 'm-', linewidth=2)
#plt.axhline(angle_threshold, color='r', linestyle='--', label=f'Порог ({angle_threshold} рад)')
plt.xlabel('Время (с)', fontsize=12)
plt.ylabel('Ошибка угла (рад)', fontsize=12)
plt.title('Изменение ошибки ориентации', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xlim(0, 4)
plt.ylim(0.07, None)
#plt.legend(fontsize=10)
plt.tight_layout()
plt.show()
