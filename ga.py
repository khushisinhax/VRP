#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import random
from random import shuffle
import re
import datetime
import logging
import json
import copy
import numpy as np

# 假设工作时长为 12 小时，即 720 分钟（从 8:00 至 20:00）
MAX_WORK_MINUTES = 720

class Generation:
    def __init__(
        self,
        aim,
        groupnum: int = 10,
        generation: int = 50,
        var_num: int = 2,
        crossrate: float = 0.8,
        variationrate: float = 0.8,
        var_minrange: list = None,
        var_maxrange: list = None,
        decodemap: dict = None,
    ):
        """
        :param aim:          适应度函数，必须返回三元组 (fitness, route, courier_spots)
        :param groupnum:     种群数量
        :param generation:   进化代数
        :param var_num:      染色体长度
        :param crossrate:    交叉概率
        :param variationrate:变异概率
        :param var_minrange: 基因最小值列表
        :param var_maxrange: 基因最大值列表
        :param decodemap:    映射字典(可选)
        """
        if var_minrange is None:
            var_minrange = [1]  # 对于扩展编码，首个基因的最小值（快递员数量）
        if var_maxrange is None:
            var_maxrange = [10]  # 对于扩展编码，首个基因的最大值
        if decodemap is None:
            decodemap = {}

        self.aim = aim
        self.groupnum = groupnum
        self.var_num = var_num
        self.generation = generation
        self.curiter = 1
        self.crossrate = crossrate
        self.variationrate = variationrate
        self.logger = logging.getLogger(__name__)
        self.var_minrange = var_minrange
        self.var_maxrange = var_maxrange
        self.decodemap = decodemap

        # 初始化种群：
        # 若 (var_num-1) 能被 3 整除，则认为采用扩展编码（1+3×N 的结构）
        self.population = []
        if (self.var_num - 1) % 3 == 0:
            N = (self.var_num - 1) // 3
            for _ in range(groupnum):
                # 基因0：快递员数量
                courier = random.randint(self.var_minrange[0], self.var_maxrange[0])
                # 基因1～N：配送顺序（对停靠点编号 1～N 的排列）
                permutation = list(range(1, N + 1))
                shuffle(permutation)
                # 基因 N+1～2N：候选到达时间（0～MAX_WORK_MINUTES 内随机）
                candidate_arrival = [random.randint(0, MAX_WORK_MINUTES) for _ in range(N)]
                # 基因 2N+1～3N：候选离开时间（保证至少比对应到达时间大 5 分钟）
                candidate_departure = []
                for arr in candidate_arrival:
                    dep = arr + random.randint(5, 30)
                    if dep > MAX_WORK_MINUTES:
                        dep = MAX_WORK_MINUTES
                    candidate_departure.append(dep)
                chromosome = [courier] + permutation + candidate_arrival + candidate_departure
                self.population.append(chromosome)
        else:
            # 原有的简单初始化（仅适用于纯排列编码）
            init_list = list(range(1, self.var_num))
            for _ in range(groupnum):
                shuffle(init_list)
                p_tmp = [random.randint(self.var_minrange[0], self.var_maxrange[0])] + copy.copy(init_list)
                self.population.append(p_tmp)

        # self.best 用于记录最优个体，格式：(best_fitness, best_chromosome, best_courier_info)
        self.best = []

    def repair_permutation(self, perm, N):
        """
        修复排列 perm，使之为 1~N 的合法排列
        """
        missing = set(range(1, N+1)) - set(perm)
        seen = set()
        for i in range(len(perm)):
            if perm[i] in seen:
                if missing:
                    perm[i] = missing.pop()
                    seen.add(perm[i])
                else:
                    # 万一 missing 为空，设为1
                    perm[i] = 1
            else:
                seen.add(perm[i])
        # 若还缺失数字，则依次补充
        for num in range(1, N+1):
            if num not in seen:
                # 替换第一个重复的元素
                for i in range(len(perm)):
                    if perm.count(perm[i]) > 1:
                        perm[i] = num
                        seen.add(num)
                        break
        return perm

    def geneDecode(self, pop: list) -> list:
        """
        如果需要位编码->浮点值转换，可在这里实现。
        当前示例未使用二进制编码，保留空实现。
        """
        return []

    def calcSufficiency(self) -> list:
        """
        计算适应度：对每个个体 pop，调用 self.aim(pop) 并做三元拆包。
        然后记录最优个体到 self.best。
        返回存活率列表(轮盘赌用)。
        """
        survival_list = []
        route_list = []
        courier_spots_list = []

        for individual in self.population:
            # aim(individual) 返回三元组 (fitness, route, courier_spots)
            fitness, route, courier_spots = self.aim(individual)
            survival_list.append(fitness)
            route_list.append(route)
            courier_spots_list.append(courier_spots)

        total = float(sum(survival_list)) if survival_list else 1.0
        rate_survival_list = [rate / total for rate in survival_list]

        # 找到适应度最高的个体下标
        index = np.argsort(rate_survival_list)[-1]

        # 记录最优个体到 self.best
        self.best.append(
            (
                survival_list[index],
                copy.copy(self.population[index]),
                courier_spots_list[index]
            )
        )

        self.curiter += 1
        return rate_survival_list

    def choosePopulation(self) -> list:
        """
        轮盘赌选择，基于存活率选出新的种群
        """
        survival_list = self.calcSufficiency()
        # 累加
        for i in range(1, len(survival_list)):
            survival_list[i] += survival_list[i - 1]

        new_population = []
        for _ in range(self.groupnum):
            random_rate = random.random()
            for i, prop in enumerate(survival_list):
                if random_rate <= prop:
                    new_population.append(copy.copy(self.population[i]))
                    break
        self.population = new_population
        return new_population

    def crossCalc(self) -> list:
        """
        交叉操作
        """
        self.choosePopulation()
        np.random.shuffle(self.population)
        if (self.var_num - 1) % 3 == 0:
            N = (self.var_num - 1) // 3
            for i in range(0, self.groupnum - 1, 2):
                # 对快递员数量基因 (索引0)
                if random.random() <= self.crossrate:
                    self.population[i][0], self.population[i+1][0] = self.population[i+1][0], self.population[i][0]
                # 对配送顺序部分：基因1～N
                rand_cross_point = random.randint(2, N)
                temp1 = self.population[i][1:rand_cross_point+1]
                temp2 = self.population[i+1][1:rand_cross_point+1]
                self.population[i][1:rand_cross_point+1] = temp2
                self.population[i+1][1:rand_cross_point+1] = temp1
                # 修复配送顺序部分，保证为合法排列
                self.population[i][1:1+N] = self.repair_permutation(self.population[i][1:1+N], N)
                self.population[i+1][1:1+N] = self.repair_permutation(self.population[i+1][1:1+N], N)
                # 对候选到达部分：基因 N+1～2N
                for j in range(N+1, 2*N+1):
                    if random.random() <= self.crossrate:
                        self.population[i][j], self.population[i+1][j] = self.population[i+1][j], self.population[i][j]
                # 对候选离开部分：基因 2N+1～3N
                for j in range(2*N+1, 3*N+1):
                    if random.random() <= self.crossrate:
                        self.population[i][j], self.population[i+1][j] = self.population[i+1][j], self.population[i][j]
        else:
            # 原有交叉方法：仅对除第一个基因外的排列进行交叉
            for i in range(0, self.groupnum, 2):
                prop = random.random()
                rand_cross_point = random.randint(2, self.var_num - 1)
                if prop <= self.crossrate:
                    i_set = set(self.population[i+1][1:rand_cross_point])
                    total_set = set(range(1, self.var_num))
                    ip1_set = set(self.population[i][1:rand_cross_point])

                    self.population[i][1:rand_cross_point], self.population[i+1][1:rand_cross_point] = (
                        copy.copy(self.population[i+1][1:rand_cross_point]),
                        copy.copy(self.population[i][1:rand_cross_point]),
                    )
                    for leftpart in range(rand_cross_point, self.var_num):
                        if self.population[i][leftpart] in i_set:
                            cur_set = total_set - i_set
                            self.population[i][leftpart] = list(cur_set)[random.randint(0, len(cur_set) - 1)]
                        i_set.add(self.population[i][leftpart])

                        if self.population[i+1][leftpart] in ip1_set:
                            cur_set = total_set - ip1_set
                            self.population[i+1][leftpart] = list(cur_set)[random.randint(0, len(cur_set) - 1)]
                        ip1_set.add(self.population[i+1][leftpart])
        return self.population

    def geneRevolution(self) -> list:
        """
        基因突变
        """
        self.crossCalc()
        if (self.var_num - 1) % 3 == 0:
            N = (self.var_num - 1) // 3
            for i in range(self.groupnum):
                # 对快递员数量基因突变
                if random.random() <= self.variationrate:
                    self.population[i][0] = random.randint(self.var_minrange[0], self.var_maxrange[0])
                # 对配送顺序部分：基因1～N
                for _ in range(random.randint(1, N)):
                    if random.random() <= self.variationrate:
                        p1 = random.randint(1, N)
                        p2 = random.randint(1, N)
                        while p2 == p1:
                            p2 = random.randint(1, N)
                        self.population[i][p1], self.population[i][p2] = self.population[i][p2], self.population[i][p1]
                # 修复配送顺序部分
                self.population[i][1:1+N] = self.repair_permutation(self.population[i][1:1+N], N)
                # 对候选到达部分：基因 N+1～2N
                for j in range(N+1, 2*N+1):
                    if random.random() <= self.variationrate:
                        self.population[i][j] = random.randint(0, MAX_WORK_MINUTES)
                # 对候选离开部分：基因 2N+1～3N
                for j in range(2*N+1, 3*N+1):
                    if random.random() <= self.variationrate:
                        arr_time = self.population[i][j - N]  # 对应的到达时间
                        self.population[i][j] = random.randint(arr_time + 5, MAX_WORK_MINUTES) if arr_time + 5 <= MAX_WORK_MINUTES else MAX_WORK_MINUTES
        else:
            # 原有突变：对第一个基因和剩余排列随机交换
            for i in range(self.groupnum):
                rand_variation_num = random.randint(1, self.var_num)
                if random.random() <= self.variationrate:
                    self.population[i][0] = random.randint(self.var_minrange[0], self.var_maxrange[0])
                for _ in range(rand_variation_num):
                    if random.random() <= self.variationrate:
                        rand_variation_point1 = random.randint(1, self.var_num - 1)
                        while True:
                            rand_variation_point2 = random.randint(1, self.var_num - 1)
                            if rand_variation_point2 != rand_variation_point1:
                                break
                        self.population[i][rand_variation_point1], self.population[i][rand_variation_point2] = (
                            self.population[i][rand_variation_point2],
                            self.population[i][rand_variation_point1]
                        )
        return self.population

    def geneEvolve(self):
        """
        多代进化后，返回最优个体。
        self.best[-1] = (best_fitness, best_chromosome, best_courier_info)
        """
        for _ in range(self.generation):
            self.geneRevolution()
        self.best.sort(key=lambda kk: kk[0])
        return self.best[-1]

if __name__ == '__main__':
    # 示例 aim 函数，返回三元组 (fitness, route, courier_spots)
    def example_aim(pop):
        # 对扩展编码结构打印各部分信息
        if (len(pop) - 1) % 3 == 0:
            N = (len(pop) - 1) // 3
            courier = pop[0]
            permutation = pop[1:1+N]
            arrival = pop[1+N:1+2*N]
            departure = pop[1+2*N:1+3*N]
            fitness = sum(k**2 for k in permutation) + courier  # 示例计算
            courier_spots = {"courier": courier, "permutation": permutation, "arrival": arrival, "departure": departure}
            route = permutation
            return fitness, route, courier_spots
        else:
            fitness = sum(k**2 for k in pop)
            route = pop[1:]
            courier_spots = {}
            return fitness, route, courier_spots

    # 示例中设定 var_num 为 1+3*N, 如 N=5
    chromo_length = 1 + 3 * 5
    g = Generation(
        example_aim,
        groupnum=10,
        generation=5,
        var_num=chromo_length,
        var_minrange=[1],
        var_maxrange=[10],
        decodemap={}
    )
    best_result = g.geneEvolve()
    print(best_result)
