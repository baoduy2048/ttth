import numpy as np
import random
import time
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm 
import math
import typing 
from matplotlib import patches as ptc 

# HÀM HỖ TRỢ

def is_within_FoV(bisector, target, sensor, radius): # 
    """
    Hàm này được điều chỉnh từ 'is_covered' để khớp với 'init_T'.
    """
    target = np.asarray(target)
    sensor = np.asarray(sensor)
    bisector = np.asarray(bisector)
    v = target - sensor
    dist = np.linalg.norm(v)
    if dist <= 1e-12:  # target exactly at sensor
        return True
    scalar = bisector.dot(v)
    # FoV half-angle pi/8
    return scalar + 1e-7 >= radius*dist*np.cos(np.pi/8) and dist - 1e-7 <= radius

# Khởi tạo ma trận T
def init_T(network): # [cite: 34-52]
    sensors = network['sensors']
    targets = network['targets']
    radius = network['radius']
    n = network['n']
    m = network['m']
    q = network['q']
    T = np.zeros((n, m, q), dtype=bool)

    bisectors = []
    for i in range(q):
        bisectors.append((radius*np.cos(np.pi*(1 + i*2)/q), radius*np.sin(np.pi*(1 + i*2)/q)))

    for i in range(n):
        for j in range(m):
            for p in range(q):
                T[i, j, p] = is_within_FoV(bisectors[p], targets[j], sensors[i], radius)
    return T

# HÀM TÍNH TOÁN FITNESS

class FitnessCalculator:
    def __init__(self, network_data):
        self.network_data = network_data # Lưu trữ network_data
        self.sensors = network_data['sensors']
        self.targets = network_data['targets']
        self.K = network_data['K']
        self.n = network_data['n']
        self.m = network_data['m']
        self.q = network_data['q']
        self.radius = network_data['radius']
        self.T_matrix = init_T(network_data) 
        
        # Tham số phạt từ config của TaP-GA
        self.lambda_useless = 1.0 
        self.lambda_active = 1./(self.n + 1)

    def get_coverage_stats(self, particle):
        """
        Tính toán và trả về mảng achieved_coverage, số cảm biến useless và active.
        Sử dụng T_matrix đã tính toán trước để tăng tốc.
        """
        f = np.zeros((self.m,), dtype=int)
        useless = 0
        active_sensors = 0

        for i in range(self.n):
            sensor_dir = particle[i]
            if sensor_dir == self.q: # q là TẮT
                continue
            
            # Kiểm tra xem cảm biến có đang phủ mục tiêu nào không
            if np.any(self.T_matrix[i, :, sensor_dir]):
                active_sensors += 1
                # Tính tổng số mục tiêu mà cảm biến này phủ
                f += self.T_matrix[i, :, sensor_dir]
            else:
                useless += 1 # Cảm biến được bật nhưng không phủ mục tiêu nào
                
        return f, useless, active_sensors

    def calculate_fitness(self, particle):
        """
        Tính toán fitness (để GIẢM THIỂU) dựa trên Formula 8 của TaP-GA.
        """
        f, useless, active_sensors = self.get_coverage_stats(particle)
        
        k_required = np.array(self.K)
        coverage_error = np.maximum(0, k_required - f)
        weighted_error = np.sum(k_required * (coverage_error ** 2))
        
        penalty_b = self.lambda_useless * useless
        penalty_c = self.lambda_active * active_sensors
        
        fitness = weighted_error + penalty_b + penalty_c
        
        num_targets_fully_covered = np.sum(f >= k_required)
        return fitness, num_targets_fully_covered, active_sensors

# THUẬT TOÁN : DPSO_GA

class DPSO_GA_Adapted:
    """
    Triển khai Algorithm 3 (DPSO_GA) từ pso.pdf.
    """
    def __init__(self, fitness_calc, pop_size, iterations, w_prob, c1, c2, pc, pm):
        self.calc = fitness_calc
        self.pop_size = pop_size
        self.iterations = iterations
        self.w_prob = w_prob
        self.c1 = c1
        self.c2 = c2
        self.pc = pc
        self.pm = pm
        self.n_sensors = self.calc.n
        self.q_plus_1 = self.calc.q + 1 

        self.population = [self._create_random_particle() for _ in range(pop_size)]
        self.fitness = [self.calc.calculate_fitness(p)[0] for p in self.population]
        
        self.pBest_particles = np.copy(self.population)
        self.pBest_fitness = np.copy(self.fitness)
        
        best_idx = np.argmin(self.fitness)
        self.gBest_particle = self.population[best_idx].copy()
        self.gBest_fitness = self.fitness[best_idx]

    def _create_random_particle(self):
        return np.random.randint(0, self.q_plus_1, size=self.n_sensors)

    # Các toán tử của Giai đoạn DPSO
    def _operator_F1_mutate(self, particle):
        temp_particle = particle.copy()
        for i in range(self.n_sensors):
            if random.random() < self.w_prob:
                temp_particle[i] = random.randint(0, self.q_plus_1 - 1)
        return temp_particle
        
    def _operator_F2_crossover(self, particle, pBest):
        temp_particle = particle.copy()
        for i in range(self.n_sensors):
            if random.random() < self.c1:
                temp_particle[i] = pBest[i]
        return temp_particle

    def _operator_F3_crossover(self, particle, gBest):
        new_particle = particle.copy()
        for i in range(self.n_sensors):
            if random.random() < self.c2:
                new_particle[i] = gBest[i]
        return new_particle

    # Các toán tử của Giai đoạn GA
    def _selection_roulette(self):
        inverted_fitness = 1.0 / (np.array(self.fitness) + 1e-9)
        total_fit = np.sum(inverted_fitness)
        
        if total_fit == 0:
            return self.population[random.randint(0, self.pop_size - 1)].copy()
            
        pick = random.uniform(0, total_fit)
        current = 0
        for i in range(self.pop_size):
            current += inverted_fitness[i]
            if current > pick:
                return self.population[i].copy()
        return self.population[-1].copy()

    def _crossover_uniform(self, p1, p2):
        if random.random() > self.pc:
            return p1.copy(), p2.copy()
        
        mask = np.random.randint(0, 2, size=self.n_sensors)
        c1 = np.where(mask == 1, p1, p2)
        c2 = np.where(mask == 1, p2, p1)
        return c1, c2

    def _mutation_random(self, particle):
        for i in range(self.n_sensors):
            if random.random() < self.pm:
                particle[i] = random.randint(0, self.q_plus_1 - 1)
        return particle

    def run(self, verbose=0):
        history = []
        
        iterable = range(self.iterations)
        if verbose > 0:
            iterable = tqdm(range(self.iterations), desc="Đang chạy DPSO-GA")

        for _ in iterable:
            
            # Giai đoạn 1: DPSO
            intermediate_population = []
            intermediate_fitness = []
            for i in range(self.pop_size):
                particle = self.population[i]
                temp_p1 = self._operator_F1_mutate(particle)
                temp_p2 = self._operator_F2_crossover(temp_p1, self.pBest_particles[i])
                new_particle = self._operator_F3_crossover(temp_p2, self.gBest_particle)
                
                new_fitness, _, _ = self.calc.calculate_fitness(new_particle)
                
                if new_fitness < self.pBest_fitness[i]:
                    self.pBest_fitness[i] = new_fitness
                    self.pBest_particles[i] = new_particle.copy()
                if new_fitness < self.gBest_fitness:
                    self.gBest_fitness = new_fitness
                    self.gBest_particle = new_particle.copy()
                    
                intermediate_population.append(new_particle)
                intermediate_fitness.append(new_fitness)
            
            self.population = intermediate_population
            self.fitness = intermediate_fitness

            # Giai đoạn 2: GA
            new_population = []
            best_idx = np.argmin(self.fitness)
            new_population.append(self.population[best_idx].copy()) # Elitism
            
            while len(new_population) < self.pop_size:
                p1 = self._selection_roulette()
                p2 = self._selection_roulette()
                c1, c2 = self._crossover_uniform(p1, p2)
                c1 = self._mutation_random(c1)
                c2 = self._mutation_random(c2)
                
                new_population.append(c1)
                if len(new_population) < self.pop_size:
                    new_population.append(c2)
            
            self.population = new_population
            self.fitness = [self.calc.calculate_fitness(p)[0] for p in self.population]
            
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.gBest_fitness:
                self.gBest_fitness = self.fitness[best_idx]
                self.gBest_particle = self.population[best_idx].copy()
                
            history.append(self.gBest_fitness)
            
        return self.gBest_particle, self.gBest_fitness, history

# CÁC HÀM METRICS

def distance_index(k, x): 
  k_sq_sum = np.sum(k*k)
  if k_sq_sum == 0: return 1.0 
  b = k - x
  b_sq_sum = np.sum(b*b)
  return 1 - (b_sq_sum / k_sq_sum)

def variance(k, x):
  m = len(x)
  if m == 0: return 0.0
  mk = np.zeros_like(x, dtype=float)
  for t in range(m):
    mk[t] = np.sum(k == k[t])
    
  nu_k = np.zeros_like(x, dtype=float)
  for t in range(m):
    ans = 0
    for i in range(m):
      ans += x[i]*(k[i] == k[t])
    if mk[t] > 0:
        nu_k[t] = ans/mk[t]
    else:
        nu_k[t] = 0

  a = (x - nu_k)
  with np.errstate(divide='ignore', invalid='ignore'):
    var_terms = (a*a) / mk
    var_terms[~np.isfinite(var_terms)] = 0 
  return np.sum(var_terms)

def activated_sensors(genome, bound): 
  cnt = 0
  for i in genome:
    if i != bound:
      cnt += 1
  return cnt

def coverage_quality(mask, network, T_matrix): 
  sensors = network['sensors']
  targets = network['targets']
  radius = network['radius']
  n = network['n']
  m = network['m']
  q = network['q']
  
  if radius == 0: return 0.0 

  U = np.zeros((n, q, m), dtype=float)
  for i in range(n):
    sensor_pos = np.asarray(sensors[i])
    for j in range(m):
        target_pos = np.asarray(targets[j])
        v = target_pos - sensor_pos
        dist_sq = np.dot(v, v)
        
        for p in range(q):
          if T_matrix[i, j, p]:
            U[i, p, j] = 1 - (dist_sq / (radius**2))

  S = np.zeros((n, q), dtype=bool)
  for i in range(n):
    if mask[i] != q:
      S[i, mask[i]] = True

  return np.sum(np.sum(U, axis=2) * S)

# HÀM MAIN

if __name__ == "__main__":
    
    # 1. Tải Dữ liệu
    print("Đang tải dữ liệu từ fix_targets.pkl...")
    try:
        with open("fix_targets.pkl", 'rb') as f:
            ft_data = pickle.load(f)
    except FileNotFoundError:
        print("LỖI: Không tìm thấy file 'fix_targets.pkl'.")
        print("Vui lòng chạy file 'Dataset.ipynb' trước để tạo file dữ liệu.")
        exit()

    # 2. Thiết lập các tham số
    MAX_GENS = 100 
    POP_SIZE = 100
    PC = 0.8
    PM = 0.1
    W_PROB = 0.1
    C1 = 0.5
    C2 = 0.5
    NUM_RUNS = 10 

    # 3. Chạy thực nghiệm cho ft_small
    DI_ft_small = []
    VAR_ft_small = []
    CQ_ft_small = []
    ACT_ft_small = []
    
    start_batch_time = time.time()

    for i in range(NUM_RUNS):
        di, var, cq, act = [], [], [], []
        
        dataset_instances = ft_data[i]['small']
        
        for dt in tqdm(dataset_instances, desc=f"Dataset {i+1}/{NUM_RUNS} (Small)"):
            
            fitness_calc = FitnessCalculator(dt)
            
            np.random.seed(i) 
            random.seed(i)

            solver = DPSO_GA_Adapted(
                fitness_calc, POP_SIZE, MAX_GENS, W_PROB, C1, C2, PC, PM
            )
            best_genome, best_fitness, history = solver.run(verbose=0)
            
            # Tính toán metrics
            f, useless, active = fitness_calc.get_coverage_stats(best_genome)
            
            DI_score = distance_index(np.asarray(dt['K']), f)
            var_score = variance(np.asarray(dt['K']), f)
            # Truyền T_matrix đã được tính vào hàm metric
            cq_score = coverage_quality(best_genome, dt, fitness_calc.T_matrix) 
            act_score = activated_sensors(best_genome, dt['q'])
            
            di.append(DI_score)
            var.append(var_score)
            cq.append(cq_score)
            act.append(act_score)
            
        DI_ft_small.append(di)
        VAR_ft_small.append(var)
        CQ_ft_small.append(cq)
        ACT_ft_small.append(act)

    DPSO_GA_ft_small = [DI_ft_small, VAR_ft_small, CQ_ft_small, ACT_ft_small]
    with open("DPSO_GA_ft_small.pkl", 'wb') as f:
        pickle.dump(DPSO_GA_ft_small, f)
    print(f"Đã lưu kết quả vào 'DPSO_GA_ft_small.pkl'")

    # 4. Chạy thực nghiệm cho ft_large
    DI_ft_large = []
    VAR_ft_large = []
    CQ_ft_large = []
    ACT_ft_large = []

    for i in range(NUM_RUNS):
        di, var, cq, act = [], [], [], []
        
        dataset_instances = ft_data[i]['large']
        for dt in tqdm(dataset_instances, desc=f"Dataset {i+1}/{NUM_RUNS} (Large)"):
            
            fitness_calc = FitnessCalculator(dt)
            
            np.random.seed(i)
            random.seed(i)

            solver = DPSO_GA_Adapted(
                fitness_calc, POP_SIZE, MAX_GENS, W_PROB, C1, C2, PC, PM
            )
            best_genome, best_fitness, history = solver.run(verbose=0)
            
            f, useless, active = fitness_calc.get_coverage_stats(best_genome)
            
            DI_score = distance_index(np.asarray(dt['K']), f)
            var_score = variance(np.asarray(dt['K']), f)
            cq_score = coverage_quality(best_genome, dt, fitness_calc.T_matrix)
            act_score = activated_sensors(best_genome, dt['q'])
            
            di.append(DI_score)
            var.append(var_score)
            cq.append(cq_score)
            act.append(act_score)
            
        DI_ft_large.append(di)
        VAR_ft_large.append(var)
        CQ_ft_large.append(cq)
        ACT_ft_large.append(act)

    DPSO_GA_ft_large = [DI_ft_large, VAR_ft_large, CQ_ft_large, ACT_ft_large]
    with open("DPSO_GA_ft_large.pkl", 'wb') as f:
        pickle.dump(DPSO_GA_ft_large, f)
    print(f"Đã lưu kết quả vào 'DPSO_GA_ft_large.pkl'")
    print(f"Tổng thời gian chạy: {(time.time() - start_batch_time) / 60:.2f} phút")