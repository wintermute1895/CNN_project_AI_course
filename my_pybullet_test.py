import pybullet as p
import pybullet_data
import time

# --- 连接到物理引擎 ---
# p.GUI: 会创建一个图形用户界面窗口来显示仿真。
# p.DIRECT: 不会创建GUI窗口，在后台运行仿真（适用于服务器或不需要可视化的情况）。
physicsClient = p.connect(p.GUI) # 或者 p.DIRECT

# 添加 PyBullet 自带的数据路径，这样可以方便地加载一些预设模型
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# --- 设置仿真环境 ---
p.setGravity(0, 0, -9.81) # 设置重力
planeId = p.loadURDF("plane.urdf") # 加载一个地面

# 加载一个简单的立方体作为障碍物
cubeStartPos = [0, 0, 1]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
# 你可以创建一个简单的URDF文件来描述立方体，或者使用 p.createMultiBody
# 为了简单，我们先加载一个已有的URDF，比如鸭子（只是为了演示加载）
duckId = p.loadURDF("duck_vhacd.urdf", cubeStartPos, cubeStartOrientation)
# 或者创建一个简单的形状
# boxHalfExtents = [0.5, 0.5, 0.5]
# colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=boxHalfExtents)
# visualShapeId = p.createVisualShape(p.GEOM_BOX, halfExtents=boxHalfExtents, rgbaColor=[1,0,0,1])
# obstacleId = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=colBoxId, baseVisualShapeIndex=visualShapeId, basePosition=[0,0,1])


# --- 仿真循环 ---
try:
    for i in range(10000): # 运行一段时间
        p.stepSimulation() # 向前推进一步仿真
        time.sleep(1./240.) # 控制仿真速度，大约240Hz的物理步长

        # 在这里可以添加获取相机图像、控制机器人等逻辑
        # 例如，获取鸭子的位置和姿态
        if i % 100 == 0: # 每100步打印一次
            duckPos, duckOrn = p.getBasePositionAndOrientation(duckId)
            print(f"Step {i}: Duck position: {duckPos}, Duck orientation: {duckOrn}")

except KeyboardInterrupt:
    print("Simulation stopped by user.")
finally:
    # --- 断开与物理引擎的连接 ---
    p.disconnect()