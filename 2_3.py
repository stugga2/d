import pybullet as p
import pybullet_data
import math
import time
import matplotlib.pyplot as plt
import numpy as np

# симуляция
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setRealTimeSimulation(0)
p.loadURDF("plane.urdf")
plane_id = p.loadURDF("plane.urdf")

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

wall_1 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1.5, 1/4, 1/4])
p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_1, basePosition=[-1.5, -3.5, 1/4])
obstacle_id = wall_1

wall_2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1/4, 3.5, 1/4])
p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_2, basePosition=[-1.5, 1.5, 1/4])
obstacle_id2 = wall_2

# обратная кинематика для модели трёхколёсного омниробота
def get_wheel_velocities(vx, vy, wz):
    wheel_vels = []
    for theta in angles:
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

# цикл симуляции

target_pos = [-4.0, -4.0]
kp_linear = 0.8
kp_yaw = 1.8
max_speed = 0.3
max_wz = 1.
stop_threshold = 0.05

# настройка параметров лучей
NUM_RAYS = 30                   # количество лучей
RAY_LENGTH = 3.0                # длина одного луча
RAYS_ANGLE = math.radians(348)  # общий угол
ANGLE_STEP = RAYS_ANGLE / (NUM_RAYS - 1) if NUM_RAYS > 1 else 0

LASER_OFFSET = [0, 0, 0.1]

last_ray_time = time.time()
RAY_CHECK_INTERVAL = 0.05  # интервал для проверки лучей

debug_line_ids = []

obstacle=False
current_target_pos=target_pos
temp_target_pos=target_pos


# массивы для записи истории (для графиков)
time_hist = []
pos_hist = []      # [x, y, yaw]
desired_hist = []  # целевые точки
step_time = 0.0
dt_sim = 1./240.
start_time = time.time()

# начальная позиция
start_pos_xy = [0.0, 0.0]

for step in range(7000):
    ###
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    x, y = pos[0], pos[1]
    _, _, yaw = p.getEulerFromQuaternion(orn)
    
    if obstacle:
        current_target_pos=temp_target_pos
    else:
        current_target_pos=target_pos
    
    dx = x - current_target_pos[0]
    dy = y - current_target_pos[1]
    distance = math.sqrt(dx*dx + dy*dy)


    # запись данных для графиков
    time_hist.append(step_time)
    pos_hist.append([x, y, yaw])
    
    # желаемая позиция (текущая цель с учётом обхода препятствий)
    desired_hist.append([current_target_pos[0], current_target_pos[1], 0.0])

    
    if distance < stop_threshold:
        vx_world, vy_world, wz = 0.0, 0.0, 0.0
        for i in range(3):
            p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, targetVelocity=0.0, force=20.0)
        p.stepSimulation()
        print(f"({current_target_pos[0]}, {current_target_pos[1]})")
        break
    else:
        angle_to_target = math.atan2(dy, dx)
        yaw_error = angle_to_target - yaw
        
        yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))
        
        vx_world = max(-max_speed, min(max_speed, kp_linear * dx))
        vy_world = max(-max_speed, min(max_speed, kp_linear * dy))
        wz = max(-max_wz, min(max_wz, kp_yaw * yaw_error))

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
        
    if step % 450 == 0:
        new_pos = [pos[0], pos[1], pos[2]]
        p.resetBasePositionAndOrientation(robot_id, new_pos, orn)
        
        # rayTestBatch с заданным интервалом
        current_time = time.time()
        if current_time - last_ray_time >= RAY_CHECK_INTERVAL:
            
            # получаем позицию и ориентацию робота
            robot_pos, robot_orn = p.getBasePositionAndOrientation(robot_id)
            
            rot_matrix = np.array(p.getMatrixFromQuaternion(robot_orn)).reshape(3, 3)
            
            # списки для rayTestBatch
            ray_from_list = []
            ray_to_list = []
            ray_colors = []
            
            for i in range(NUM_RAYS):
                
                # вычисляем угол для текущего луча
                if NUM_RAYS > 1:
                    angle = -RAYS_ANGLE / 2 + i * ANGLE_STEP
                else:
                    angle = 0
                
                # направление луча в локальной системе координат робота
                local_dir = np.array([math.cos(angle), math.sin(angle), 0.0])
                
                # направление в глобальной системе координат
                world_dir = rot_matrix.dot(local_dir)
                world_dir = world_dir / np.linalg.norm(world_dir)  # нормализуем
                
                # начальная точка луча в глобальной системе координат
                start_pos = np.array(robot_pos) + np.array(LASER_OFFSET)
                
                # конечная точка луча
                end_pos = start_pos + world_dir * RAY_LENGTH
                
                ray_from_list.append(start_pos)
                ray_to_list.append(end_pos)
                ray_colors.append([1, 0, 0])  # красный
                
            hit_results = p.rayTestBatch(ray_from_list, ray_to_list)
            
            print(f"\nРезультаты rayTestBatch в момент времени: {current_time:.2f}")

            mas=[]
            
            for i, hit in enumerate(hit_results):
                hit_object_id = hit[0]
                hit_fraction = hit[2]

                mas.append([])
                
                if hit_object_id != -1:
                    
                    # направление луча в глобальной системе координат
                    ray_dir = ray_to_list[i] - ray_from_list[i]
                    ray_dir = ray_dir / np.linalg.norm(ray_dir)
                    
                    # вектор от модели робота к target_pos
                    target_world = np.array([target_pos[0], target_pos[1], robot_pos[2]])
                    to_target = target_world - np.array(robot_pos)
                    to_target = to_target / np.linalg.norm(to_target)
                    
                    # угол между лучом и направлением на target_pos
                    dot_product = np.clip(np.dot(ray_dir, to_target), -1.0, 1.0)
                    angle_rad = math.acos(dot_product)
                    angle_deg = math.degrees(angle_rad)
                    
                    print(f"  Луч {i}, {hit_object_id}, расстояние: {hit_fraction * RAY_LENGTH:.2f} м, угол до цели: {angle_deg:.1f}°")
                    
                    ray_dir_2d = np.array([ray_dir[0], ray_dir[1]])
                    to_target_2d = np.array([to_target[0], to_target[1]])
                    
                    cross_z = ray_dir_2d[0] * to_target_2d[1] - ray_dir_2d[1] * to_target_2d[0]
                    
                    # определяем направление
                    if cross_z > 0:
                        direction_sign = +1  # против часовой стрелки
                    else:
                        direction_sign = -1  # по часовой стрелке

                    mas[i].extend([i])
                    mas[i].extend([hit_object_id])
                    mas[i].extend([hit_fraction * RAY_LENGTH])
                    mas[i].extend([angle_deg])
                    mas[i].extend([direction_sign])
                    
            mas2=[]
            for i1 in mas:
                if i1:
                    mas2.append(i1)
            
            min_distance = min(item[2] for item in mas2)
            for i2 in mas2:
                if i2[2] == min_distance:
                    min_hit_object_id = i2[1]
            ray_obstacle=[]
            for i3 in mas2:
                if i3[1] == min_hit_object_id:
                    ray_obstacle.append (i3)
            
            split_index = 0
            for i5 in range(1, len(ray_obstacle)):
                if abs(ray_obstacle[i5][0] - ray_obstacle[i5-1][0]) > 1:
                    split_index = i5
                    break
                
            ray_obstacle_sorted=ray_obstacle[split_index:]+ray_obstacle[:split_index]
            
            point1=ray_obstacle_sorted[0]
            point2=ray_obstacle_sorted[-1]
            direction_sign1=point1[-1]
            direction_sign2=point2[-1]
            widthr=0.5
            
            if (point1[2] >= widthr) and (point2[2] >= widthr) and (math.hypot(x-target_pos[0],y-target_pos[1]) > 1.5) and ((direction_sign1 * direction_sign2 == -1) or (math.sin(math.radians(point1[3]))*point1[2] < 2*widthr) or (math.sin(math.radians(point2[3]))*point2[2] < 2*widthr)):
                if direction_sign1 == 1:
                    way1=math.degrees(math.asin(widthr / point1[2])) + point1[3]
                else:
                    way1=-math.degrees(math.asin(widthr / point1[2])) + point1[3]
                if direction_sign2 == 1:
                    way2=-math.degrees(math.asin(widthr / point2[2])) + point2[3]
                else:
                    way2=math.degrees(math.asin(widthr / point2[2])) + point2[3]
                if way1 < way2:
                    minway = way1*direction_sign1
                    minras = point1[2]+1
                else:
                    minway = way2*direction_sign2
                    minras = point2[2]+1
                
                pos, orn = p.getBasePositionAndOrientation(robot_id)
                _, _, yaw = p.getEulerFromQuaternion(orn)
                
                # синий луч от робота в направлении minway
                current_x, current_y = pos[0], pos[1]
                current_z = pos[2] + 0.05
                
                # minway в радианы
                relative_angle_rad = math.radians(minway)
                
                current_target_pos = target_pos
                dx = x - current_target_pos[0]
                dy = y - current_target_pos[1]
                distance = math.sqrt(dx*dx + dy*dy)
                angle_to_target = math.atan2(dy, dx)
                
                # переводим в глобальную систему координат
                if angle_to_target > 0:
                    abs_angle = - math.pi + angle_to_target - relative_angle_rad
                else: 
                    abs_angle = angle_to_target - relative_angle_rad
                    
                ray_length = minras
                end_x = current_x + ray_length * math.cos(abs_angle)
                end_y = current_y + ray_length * math.sin(abs_angle)
                
                obstacle=True
                temp_target_pos=[end_x, end_y]
                
                # синий луч
                p.addUserDebugLine(
                    [current_x, current_y, current_z],
                    [end_x, end_y, current_z],
                    [0, 0, 1],  # синий
                    3,          # толщина
                    lifeTime=0.3
                )
                
            else:
                obstacle=False
            
            # удаляем старые отладочные линии
            for line_id in debug_line_ids:
                p.removeUserDebugItem(line_id)
            debug_line_ids.clear()
            
            # новые отладочные линии
            for i, hit in enumerate(hit_results):
                if hit[0] != -1:
                    hit_fraction = hit[2]
                    hit_position = hit[3]
                    # если есть попадание рисуем зелёную линию до точки попадания
                    line_id = p.addUserDebugLine(ray_from_list[i], 
                                                  hit_position, 
                                                  [0, 1, 0],    # зелёный для попаданий
                                                  2)            # толщина
                else:
                    # если попадания нет, то рисуем красную линию
                    line_id = p.addUserDebugLine(ray_from_list[i], 
                                                  ray_to_list[i], 
                                                  [1, 0, 0],  # красный для промахов
                                                  2)
                debug_line_ids.append(line_id)
                
            last_ray_time = current_time

    p.stepSimulation()
    step_time += dt_sim
    time.sleep(1./240.)

    if step % 250 == 0:
        pos, orn = p.getBasePositionAndOrientation(robot_id)
        _, _, yaw = p.getEulerFromQuaternion(orn)
        print(f"Step {step:4d} | Pos: ({pos[0]:.2f}, {pos[1]:.2f}) | Yaw: {math.degrees(yaw):.1f}°")


# преобразование в numpy массивы
time_hist = np.array(time_hist)
pos_hist = np.array(pos_hist)

# убираем скачки при переходе через ±180°
yaw_unwrapped = np.unwrap(pos_hist[:,2])
yaw_deg = np.degrees(yaw_unwrapped)

# вычисляем ошибку позиции
error_dist = np.sqrt((pos_hist[:,0] - target_pos[0])**2 + (pos_hist[:,1] - target_pos[1])**2)

# XY траектория
plt.figure("XY траектория", figsize=(10, 8))
plt.plot(pos_hist[:,0], pos_hist[:,1], 'b-', linewidth=2, label='Траектория')
plt.plot([target_pos[0]], [target_pos[1]], 'go', markersize=10, label='Конечная цель')
plt.plot([start_pos_xy[0]], [start_pos_xy[1]], 'ro', markersize=8, label='Стартовая точка')

green_rect = plt.Rectangle((-1.75, -2), 0.5, 7, color='g', alpha=1, label='Препятствие 1')
plt.gca().add_patch(green_rect)
blue_rect = plt.Rectangle((-3, -3.75), 3, 0.5, color='b', alpha=1, label='Препятствие 2')
plt.gca().add_patch(blue_rect)

plt.xlabel('X (м)', fontsize=12)
plt.ylabel('Y (м)', fontsize=12)
plt.title('Траектория движения модели омниробота в плоскости XY с обходом препятствий', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()

# изменение положения и ориентации во времени
plt.figure("Изменение положения и ориентаци", figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(time_hist, pos_hist[:,0], 'b-', linewidth=1.5, label='x факт')
plt.ylabel('X (м)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.title('Координата X(t)', fontsize=12)

plt.subplot(3, 1, 2)
plt.plot(time_hist, pos_hist[:,1], 'b-', linewidth=1.5, label='y факт')
plt.ylabel('Y (м)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.title('Координата Y(t)', fontsize=12)

plt.subplot(3, 1, 3)
plt.plot(time_hist, yaw_deg, 'b-', linewidth=1.5, label='угол факт')
plt.xlabel('Время (с)', fontsize=12)
plt.ylabel('Угол (град)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.title('Ориентация φ(t)', fontsize=12)
y_min, y_max = yaw_deg.min(), yaw_deg.max()
y_range = y_max - y_min
plt.ylim(y_min - 0.25*y_range, y_max + 0.25*y_range)

plt.tight_layout()

plt.show()
