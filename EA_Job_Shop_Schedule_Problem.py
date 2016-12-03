import numpy as np 
import random
import matplotlib.pyplot as plt

np.random.seed(1)
jobs = []

#this is my random example, you can also self-define your jobs
population = 10
job_num = 3
machine_num = 3
operation_num = 5

#this produces an array, the frist level is each job, then for each job, the rows represent for machine
#the columns represent for operations, corresponding number represents hours it will take
for j in range(job_num):
	 job = np.random.randint(1,6,machine_num*operation_num)
	 job = job.reshape(machine_num,operation_num)
	 jobs.append(job)
jobs = np.array(jobs)
print(jobs)
#process with each job to assign machine to each operation
#the row is job, and the colunm is machine
def arrange(seed,jobs):
  jobs_Machine = []
  random.seed(seed)
  for job in jobs:
    machine = [random.randint(1,machine_num) for i in range(operation_num)]
    jobs_Machine.append(machine)
  jobs_Machine = np.array(jobs_Machine)
  return jobs_Machine

#process the population, arrange machine
def arrangePopulation(seed,jobs):
  jobs_Machines = []
  for i in range(population):
    temp = arrange(seed,jobs)
    jobs_Machines.append(temp)
  jobs_Machines = np.array(jobs_Machines)
  return jobs_Machines

def getProcessTime(jobs,job_item, operation_item,jobs_Machine):
  machine_item = jobs_Machine[job_item - 1,operation_item - 1]
  return jobs[job_item][machine_item - 1,operation_item - 1]

#calculating the finish time
def getFinishTime(jobs,jobs_Machine):
  T = np.zeros(job_num)
  D = np.zeros(machine_num)

  for i in range(operation_num):
    for j in range(job_num):
      machine = jobs_Machine[j,i]
      if T[j] < D[machine - 1]:
        temp = D[machine - 1]
      else:
        temp = T[j]

      T[j] = temp + getProcessTime(jobs,j,i,jobs_Machine)
      D[machine - 1] = temp + getProcessTime(jobs,j,i,jobs_Machine)

  return max(D)


#cross over two chromosomes
def crossover(jobs_Machines):
  a,b = random.sample(range(0,population),2)
  row = random.randint(0,machine_num-1)
  jobs_Machine1 = jobs_Machines[a] 
  jobs_Machine2 = jobs_Machines[b]
  temp1 = jobs_Machine1[row,:]
  jobs_Machine1[row,:] = jobs_Machine2[row,:]
  jobs_Machine2[row,:] = temp1
  jobs_Machines[a] = jobs_Machine1
  jobs_Machines[b] = jobs_Machine2
  #column = random.randint(0,operation_num - 1)
  #temp2 = jobs_Machine1[:,column]
  #jobs_Machine1[:,column] = jobs_Machine2[:,column]
  #jobs_Machine2[:,column] = temp2
  #jobs_Machines[a] = jobs_Machine1
  #jobs_Machines[b] = jobs_Machine2
  return jobs_Machines

def calLoad(jobs, jobs_Machine):
  Machine_Load = np.zeros(3)
  for i in range(job_num):
    for j in range(operation_num):
        machine = jobs_Machine[i,j] 
        if machine == 1:
          Machine_Load[0] = Machine_Load[0] + getProcessTime(jobs,i,j,jobs_Machine)
        elif machine == 2:
          Machine_Load[1] = Machine_Load[1] + getProcessTime(jobs,i,j,jobs_Machine)
        else:
          Machine_Load[2] = Machine_Load[2] + getProcessTime(jobs,i,j,jobs_Machine)
  index_min = np.argmin(Machine_Load)+1
  index_max = np.argmax(Machine_Load)+1
  for i in range(job_num):
    for j in range(operation_num):
       if jobs_Machine[i,j] == index_max:
         jobs_Machine[i,j] = index_min
         break
  return jobs_Machine

def mutation(jobs_Machines):
    a = random.randint(0,population-1) 
    temp = calLoad(jobs, jobs_Machines[a])
    jobs_Machines[a] = temp
    return jobs_Machines

#run genetic algorithm
seed_num = 10
max_gen = 1000
pc = 0.75
pm = 0.2
ft = np.zeros((seed_num,max_gen))
temp_ft = np.zeros(max_gen)

for seed in range(seed_num):
  np.random.seed(seed)
  jobs_Machines = arrangePopulation(seed,jobs)
  gen = 0
  f = np.zeros(population)
  f_best, x_best = None, None
  while gen < max_gen:
#get finish time
    for ix,jobs_Machine in enumerate(jobs_Machines):
      f[ix] = getFinishTime(jobs,jobs_Machine)
#keep track of best
    if f_best is None or f.min() < f_best:
      f_best = f.min()
      x_best = jobs_Machines[f.argmin()]

#crossover and mutuation
    p = random.random()
    Q = np.copy(jobs_Machines)
    if p <= pc:
      jobs_Machines = crossover(jobs_Machines) 
    if p<= pm:
      jobs_Machines = mutation(jobs_Machines)

#new population
    ft[seed,gen] = f_best
    gen += 1

  print(x_best)
  print(f_best)


plt.plot(ft.T, color = 'b')
plt.xlabel('Generations')
plt.ylabel('Makespan')
plt.show()







