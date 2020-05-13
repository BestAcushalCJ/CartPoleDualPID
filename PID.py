import gym
import numpy as np
import geatpy as ea
import time
import imageio


class plant():
    def __init__(self, timeStep):
        self.env = gym.make('CartPole-v1')
        # CartPole-v1 state说明：四个变量
        # 小车位置，小车速度，杆角度，杆角速度
        self.state = self.env.reset()
        self.timeStep = timeStep  # 一个run的最大时间步
        self.stateTrace = []
        self.imageSeq = [] # 一个run中所有帧

    def run(self, pidController):
        self.env.seed(0)
        self.state = self.env.reset()
        self.stateTrace = []
        self.imageSeq = []
        for i in range(self.timeStep):
            self.imageSeq.append(self.env.render(mode='rgb_array'))
            # 定义的控制器给出控制量 action
            self.action = pidController.mergeController(self.state)
            # 与CartPole-v1交互得到
            self.nextState, reward, done, _ = self.env.step(self.action)
            self.state = self.nextState
            self.stateTrace.append(self.state)
            if done:
                self.finishedTimeStep = i + 1
                break
            if i == self.timeStep-1:
                self.finishedTimeStep = self.timeStep

    def reward(self):
        self.stateTrace = np.array(self.stateTrace)
        self.rewardFunc = np.abs(self.stateTrace[:, [0, 2]]).sum() / self.finishedTimeStep
        self.rewardFunc += self.timeStep - self.finishedTimeStep
        return self.rewardFunc

    def close(self):
        self.env.close()

    def getGif(self):
        imageio.mimsave('cartPoleResults.gif', self.imageSeq, 'GIF', duration=0.02)

# 控制器实现
class cartPoleController:
    def __init__(self, kpCart, kiCart, kdCart, kpPole, kiPole, kdPole):
        self.kpCart, self.kiCart, self.kdCart = kpCart, kiCart, kdCart  # 小车PID参数
        self.kpPole, self.kiPole, self.kdPole = kpPole, kiPole, kdPole  # 倒立摆PID参数
        self.cartBiasLast, self.poleBiasLast = 0, 0  # bias
        self.cartBiasIntegral, self.poleBiasIntegral = 0, 0

    def cartPDController(self):
        bias = self.state[0]
        deltaBias = bias - self.cartBiasLast
        balance = self.kpCart * bias + self.kdCart * deltaBias
        self.cartBiasLast = bias
        return balance

    def polePDController(self):
        bias = self.state[2]
        deltaBias = bias - self.poleBiasLast
        balance = - self.kpPole * bias - self.kdPole * deltaBias
        self.poleBiasLast = bias
        return balance

    def cartPIDController(self):
        bias = self.state[0]
        deltaBias = bias - self.cartBiasLast
        self.cartBiasIntegral += bias
        balance = self.kpCart * bias + self.kiCart * self.cartBiasIntegral + self.kdCart * deltaBias
        self.cartBiasLast = bias
        return balance

    def polePIDController(self):
        bias = self.state[2]
        deltaBias = bias - self.poleBiasLast
        self.poleBiasIntegral += bias
        balance = - self.kpPole * bias - self.kiPole * self.cartBiasIntegral - self.kdPole * deltaBias
        self.poleBiasLast = bias
        return balance

    def mergeController(self, state):
        self.state = state
        return 1 if (self.polePIDController() - self.cartPIDController()) < 0 else 0


class optimizePID(ea.Problem):  # 继承Problem父类
    def __init__(self):
        name = 'optimizePID'
        M = 1
        maxormins = [1]  # 最小化PID函数值
        Dim = 6  # 优化的PID参数(决策变量)有6个
        varTypes = [0] * Dim # 初始化决策变量（连续）
        lb = [0, 0, 0, 0, 0, 0]  # 决策变量下界
        ub = [100, 0.1, 100, 100, 0.1, 100]  # 决策变量上界
        lbin = [0] * Dim
        ubin = [1] * Dim
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    # 目标函数，pop表示种群
    def aimFunc(self, pop):
        Vars = pop.Phen  # 决策变量矩阵
        objValues = np.zeros((NIND,1))
        for i in range(NIND):
            kpCart, kiCart, kdCart = Vars[i, [0]], Vars[i, [1]], Vars[i, [2]]
            kpPole, kiPole, kdPole = Vars[i, [3]], Vars[i, [4]], Vars[i, [5]]
            pidController = cartPoleController(kpCart, kiCart, kdCart, kpPole, kiPole, kdPole)
            cartPole.run(pidController)
            objValues[i] = cartPole.reward()
        # 将目标函数值赋值给pop的ObjV属性
        pop.ObjV = objValues


cartPole = plant(200)
problem = optimizePID()
# 种群设置
Encoding = 'RI'  # 格雷码
NIND = 30  # 种群数量
Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
population = ea.Population(Encoding, Field, NIND)
#算法参数设置
myAlgorithm = ea.soea_SEGA_templet(problem, population)
myAlgorithm.MAXGEN = 30  # 最大遗传代数
myAlgorithm.mutOper.F = 0.5  # 设置差分进化的变异缩放因子
myAlgorithm.recOper.XOVR = 0.5  # 设置交叉概率
myAlgorithm.drawing = 1  # 绘图方式

[population, obj_trace, var_trace] = myAlgorithm.run()  # 执行优化算法
cartPole.close()
best_gen = np.argmin(obj_trace[:, 1])  # 找出最优种群
best_ObjV = obj_trace[best_gen, 1]
print('最优的目标函数值为：%s'%(best_ObjV))
bestSolution = []
for i in range(var_trace.shape[1]):
    bestSolution.append(var_trace[best_gen, i])
print('最优的决策变量值为：', bestSolution)
print('有效进化代数：%s'%(obj_trace.shape[0]))
print('最优的一代是第 %s 代'%(best_gen + 1))
print('评价次数：%s'%(myAlgorithm.evalsNum))
print('时间已过 %s 秒'%(myAlgorithm.passTime))

# 得到当前进化中最优的PID参数
kpCart, kiCart, kdCart = bestSolution[0], bestSolution[1], bestSolution[2]
kpPole, kiPole, kdPole = bestSolution[3], bestSolution[4], bestSolution[5]
cartPole = plant(200)
pidController = cartPoleController(kpCart, kiCart, kdCart, kpPole, kiPole, kdPole)
cartPole.run(pidController)
print("objective values: ", cartPole.reward())
cartPole.getGif()
cartPole.close()