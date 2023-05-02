//目标函数
/* 求 21.5 + x1 * sin(4 * pi * x1) + x2 * sin(20 * pi * x2)
在-3 <= x1 <= 12.1, 4.1 <= x2 <= 5.8中的最大值 */


//头文件
#include <iostream>
#include <ctime>        //引入时间种子，计算运算时间
#include <cstdlib>      //引入库函数
#include <random>       //引入随机函数
#include <vector>       //引入数据结构存储二进制编码
#include <cmath>        //引入 sin cos acos
#include <iomanip>      //引入保留小数
#include <algorithm>    //引入shuffle函数，用于两点交叉
#include <Windows.h>    //按从上到下顺序前两个需要这个头文件所以不能跨平台只能windows
#include <sys/timeb.h>  //引入毫秒级的随机数

using namespace std;    //标准命名空间


//声明宏常量
const int ENCODE_LEN = 33;      // 编码长度
const int ENCODE_X1_LEN = 18;   // 变量1编码长度
const int ENCODE_X2_LEN = 15;   // 变量2编码长度
const int POP_SIZE = 100;       // 种群大小
const int RWS_SIZE = 20;        // 轮盘赌选择的种群大小，尽量保证为偶数，便于之后的交叉
const int E = 4;               // 交叉完选择E个直接进行保留
const int MAX_GEN = 500;        // 最大迭代次数
const double CROSS_RATE = 0.9;  // 交叉概率
const double MUTATE_RATE = 0.2; // 变异概率
const double X1_LOWER = -3;     // X1变量下界
const double X1_UPPER = 12.1;   // X1变量上界
const double X2_LOWER = 4.1;    // X2变量下界
const double X2_UPPER = 5.8;    // X2变量上界

//数学常量
#define PI acos(-1)


//目标函数
double Function(double x1, double x2) {
    return 21.5 + x1 * sin(4 * PI * x1) + x2 * sin(20 * PI * x2);
}

//随机数函数
float Random() {
    /* LARGE_INTEGER seed;
    QueryPerformanceFrequency(&seed);
    QueryPerformanceCounter(&seed); 
    srand(seed.QuadPart);//初始化一个以微秒为单位的时间种子 */
    float randnum = rand() % (999) / (float)1000;
    return randnum;
}

//个体的定义
class Individual {
public:
    vector<int> x;                      //二进制编码
    double fitness;                     //适应度
    double selectProbability;           //归一化适应度，轮盘赌选择时使用，即选择概率
    double Cumulativeprobability;       //计算累加概率

    //初始化
    Individual(){
        this->Cumulativeprobability = 0;
        this->x.resize(ENCODE_LEN);
    }
};

//种群的定义
class Population {
public:
    // Individual individuals[POP_SIZE];    //种群数组
    vector<Individual> individuals;         //种群数组
    double pop_fitness;                     //种群适应度

    //初始化
    Population() {
        this->individuals.resize(POP_SIZE);
        this->pop_fitness = 0;
    }
};

//交叉池的定义
class CrossPool {
public:
    vector<Individual> pool;
    vector<int> mark;
    double pool_fitness;

    //初始化
    CrossPool() {
        this->pool.reserve(2 * RWS_SIZE);    //容量为两倍，其会交叉生成两个子代
        // this->mark.resize(RWS_SIZE);        //标记进入交叉池的个体在种群的位置，交叉变异完后返回进行覆盖
        this->pool_fitness = 0;
    }
};


//排序函数
bool compare(const Individual indiv1, const Individual indiv2) {
    if(indiv1.fitness > indiv2.fitness)
        return true;
    else 
        return false;
}

//shuffle函数
void myShuffle(vector<Individual>::iterator begin, vector<Individual>::iterator end) {
    std::random_device rd;
    std::mt19937 rng(rd());
    shuffle (begin, end, rng);
}

/**
 * STEP0    确定编码长度
*/
//计算编码的位数
// -3.0 < x1 < 12.1 m = 18 位
//  4.1 < x2 < 5.8  m = 15 位
//因此总长度为 18 + 15 = 33 位


/**
 * STEP1    初始化种群
*/
//种群初始化
void populationInit(Population& population) {
    //使用计算机在0～1之间产生随机数K，并按照数K的
    /* 值初始化基因位：
    0 ≤ K ＜0.5 ，基因为置为1
    0.5 ≤ K ≤ 1 ，基因为置为0 */
    for(int i = 0; i < POP_SIZE; i++) {     //对其中每一个个体实现随机编码
        for(int j = 0; j < ENCODE_LEN; j++) {   //对个体的每一位进行随机编码
            /* float random = Random();
            if(random >= 0 && random <= 0.5)
                population.individuals[i].x[j] = 0;
            else if(random > 0.5  && random <= 1)
                population.individuals[i].x[j] = 1; */
                population.individuals[i].x[j] = rand() % 2;
        }
    }
}


/**
 * STEP2    计算个体适应度
*/
//先计算个体的适应度，使用引用类型返回x1和x2的值
void calculateIndividual(Individual& indiv, double& x1, double& x2) {
    //解码公式
    //y = a + (b - a) * (x / 2^m - 1)

    //先计算x1
    int ans1 = 0;   //x1的解码结果
    for(int i = 0; i < ENCODE_X1_LEN; i++) {
        if(indiv.x[i] == 1)
            ans1 += pow(2, ENCODE_X1_LEN - i - 1);
    }
    x1 = X1_LOWER + (X1_UPPER - X1_LOWER) * ((double)ans1 / (pow(2,ENCODE_X1_LEN) - 1));

    //计算x2
    int ans2 = 0;   //x2的解码结果
    for(int i = ENCODE_X1_LEN; i < ENCODE_LEN; i++) {
        if(indiv.x[i] == 1)
            ans2 += pow(2, ENCODE_LEN - i - 1);
    }
    x2 = X2_LOWER + (X2_UPPER - X2_LOWER) * ((double)ans2 / (pow(2, ENCODE_X2_LEN) - 1));
}

//计算种群中所有个体的适应度
void calculatePopulation(Population& population) {
    for(int i = 0; i < POP_SIZE; i++) {
        double x1 = 0;
        double x2 = 0;
        calculateIndividual(population.individuals[i], x1, x2);
        //计算适应度
        double fit = Function(x1, x2);
        population.individuals[i].fitness = fit;
    }
}


/**
 * STEP3    通过轮盘赌选择 N 个个体放入交叉池中
*/
//精英保留
void elitistArchive(Population& population, CrossPool& crosspool) {
    //先排序保留前 RWS 个
    // sort(population.individuals.begin(), population.individuals.end(), compare);
    for(int i = 0; i < RWS_SIZE; i++) {
        crosspool.pool.push_back(population.individuals[i]);
    }
}

//轮盘赌选择 RWS_SIZE个个体放入交叉池中
void rouletteWheelSelection(Population& population, CrossPool& crosspool) {
    //累加种群适应度
    for(int i = 0; i < POP_SIZE; i++)
        population.pop_fitness += population.individuals[i].fitness;

    //计算各个个体的选择概率
    for(int i = 0; i < POP_SIZE; i++) {
        population.individuals[i].selectProbability = population.individuals[i].fitness / population.pop_fitness;   //选择概率为 fi/fsum
    }
    //计算各个个体的累加概率
    for(int i = 0; i < POP_SIZE; i++) {
        population.individuals[i].Cumulativeprobability = 0;    //初始化
        for(int j = 0; j <= i; j++) {
            population.individuals[i].Cumulativeprobability += population.individuals[j].selectProbability; //计算累加概率
        }
    }

    
    int cnt = 0;    //计数器
    while(cnt < RWS_SIZE) {
        //生成一个0-1之间的随机小数
        float random = Random();

        //遍历选择个体放入交叉池
        
        for(int i = 0; i < POP_SIZE - 1; i++) {     //减一注意数组越界
            if(random >= 0 && random <= population.individuals[0].Cumulativeprobability) {
                crosspool.pool.push_back(population.individuals[0]);
                // crosspool.mark.push_back(0);
                cnt++;
                break;
            }else if(random > population.individuals[i].Cumulativeprobability && random <= population.individuals[i + 1].Cumulativeprobability) {
                //满足条件放入交叉池,放入第j+1个个体
                crosspool.pool.push_back(population.individuals[i + 1]);
                // crosspool.mark.push_back(i + 1);
                cnt++;
                break;
            }
        }  
    }
}


/** 
 * STEP4    根据变异概率 MUTATE_RATE 对交叉产生的个体进行变异，
 *          将单点变异作为变异算子，记产生的后代个体的集合为 O2
*/
//两点交叉函数，不使用引用，单独加入形参
void pointCrossoverAndMutate(CrossPool& crosspool, Individual& oldindiv1, Individual& oldindiv2) {
    Individual indiv1(oldindiv1);
    Individual indiv2(oldindiv2);
    int rand1 = rand() % ENCODE_X1_LEN;                     //得到一个 0 - 17 的随机数
    int rand2 = ENCODE_X1_LEN + rand() % ENCODE_X2_LEN;     //得到一个 18 - 32 的随机数
    /* myShuffle(indiv1.x.begin(), indiv1.x.end());
    myShuffle(indiv2.x.begin(), indiv2.x.end()); */
    
    //进行交换
    //对x1部分进行交叉
    for(int i = 0; i < rand1; i++) {
        int temp = indiv1.x[i];
        indiv1.x[i] = indiv2.x[i];
        indiv2.x[i] = temp;
    }
    //对x2部分进行交叉
    for(int i = ENCODE_X1_LEN; i < rand2; i++) {
        int temp = indiv1.x[i];
        indiv1.x[i] = indiv2.x[i];
        indiv2.x[i] = temp;
    }

    //对产生了交叉的个体进行变异
    for(int i = 0; i < ENCODE_LEN; i++) {   //对第一个个体进行变异
        float mutateRandom = Random();  //变异概率随机数
        if(mutateRandom <= MUTATE_RATE)
            indiv1.x[i] = (indiv1.x[i] == 1 ? 0 : 1); //位反转实现变异
    }
    for(int i = 0; i < ENCODE_LEN; i++) {   //对第二个个体进行变异
        float mutateRandom = Random();  //变异概率随机数
        if(mutateRandom <= MUTATE_RATE)
            indiv2.x[i] = (indiv2.x[i] == 1 ? 0 : 1); //位反转实现变异
    }
    //重新计算二者的fitness
    double x1 = 0;
    double x2 = 0;
    calculateIndividual(indiv1, x1, x2);
    indiv1.fitness = Function(x1, x2);
    crosspool.pool.push_back(indiv1);
    calculateIndividual(indiv2, x1, x2);
    indiv2.fitness = Function(x1, x2);
    crosspool.pool.push_back(indiv2);   //加入交叉后的子代个体
}

//精英保留交叉
void elitistArchiveCross(Population& population, CrossPool& crosspool) {
    myShuffle(crosspool.pool.begin(), crosspool.pool.end());
    for(int i = 0; i < RWS_SIZE; i+=2) {
        float crossRandom = Random();   //交叉概率随机数
        if(crossRandom <= CROSS_RATE) {  //若随机数小于等于pc则进行两点交叉，产生两个子代个体
            pointCrossoverAndMutate(crosspool, population.individuals[i], population.individuals[i + 1]);
        }
    }
}

//放入交叉池中进行两点交叉
void CrossAndMutate(Population& population, CrossPool& crosspool) {
    //首先生成 RWS_SIZE/2 个随机数进行两点交叉
    for(int i = 0; i < RWS_SIZE; i+=2) {
        float crossRandom = Random();   //交叉概率随机数
        if(crossRandom <= CROSS_RATE) {  //若随机数小于等于pc则进行两点交叉，产生两个子代个体
            pointCrossoverAndMutate(crosspool, population.individuals[i], population.individuals[i + 1]);
        }
    }
}


/**
 * STEP5    计算所有后代个体的适应度。从父代和后代的集合中选出最好的E=2个个体直接保留到下一代种群。
 *          然后，根据轮盘赌从父代和后代的集合中选出N-E=8个个体进入下一代种群。
*/
//精英保留放入
void elitistArchiveIterative(Population& population, CrossPool& crosspool) {
    sort(crosspool.pool.begin(), crosspool.pool.end(), compare);
    for(int i = 0; i < RWS_SIZE; i++) {
        population.individuals[POP_SIZE - i - 1] = crosspool.pool[i];
    }
    sort(population.individuals.begin(), population.individuals.end(), compare);
    cout << population.individuals[0].fitness << endl;
}

//轮盘赌放入
void calculateAndIterative(Population& population, CrossPool& crosspool) {
    //计算适应度
    for(int i = 0; i < crosspool.pool.size(); i++) {
        /* double x1 = 0;
        double x2 = 0;
        calculateIndividual(crosspool.pool[i], x1, x2);
        //计算适应度
        double fit = Function(x1, x2);
        crosspool.pool[i].fitness = fit; */
        crosspool.pool_fitness += crosspool.pool[i].fitness;  //累加到交叉池适应度
    }


    //由大到小进行排序
    sort(crosspool.pool.begin(), crosspool.pool.end(), compare);
    // sort(population.individuals.begin(), population.individuals.end(), compare);

    //从父代和后代的集合中选出最好的E=2个个体直接保留到下一代种群
    for(int i = 0; i < E; i++) {
        //选择原来种群位置标记的进行覆盖
        // int mark = crosspool.mark[i];
        // population.individuals[mark] = crosspool.pool[i];
        population.individuals[POP_SIZE - i - 1] = crosspool.pool[i];   //倒着覆盖最差的
        crosspool.pool_fitness -= crosspool.pool[i].fitness;    //减去前两个适应度
    }
    
    //根据轮盘赌从父代和后代的集合中选出N-E=8个个体进入下一代种群
    //计算各个个体的选择概率
    for(int i = E; i < crosspool.pool.size(); i++) {
        crosspool.pool[i].selectProbability = crosspool.pool[i].fitness / crosspool.pool_fitness;   //选择概率为 fi/fsum
    }
    //计算各个个体的累加概率
    for(int i = E; i < crosspool.pool.size(); i++) {
        crosspool.pool[i].Cumulativeprobability = 0;    //初始化
        for(int j = E; j <= i; j++) {
            crosspool.pool[i].Cumulativeprobability += crosspool.pool[j].selectProbability; //计算累加概率
        }
    }

    int cnt = E;    //计数器
    while(cnt < RWS_SIZE){
        //生成一个0-1之间的随机小数
        float random = Random();

        //遍历选择个体放入种群
        
        for(int i = E; i < crosspool.pool.size() - 1; i++) {     //减一注意数组越界
            if(random >= 0 && random <= crosspool.pool[E].Cumulativeprobability) {
                // int mark = crosspool.mark[0];
                population.individuals[POP_SIZE - cnt - 1] = crosspool.pool[E];
                cnt++;
                break;
            }
            if(random > crosspool.pool[i].Cumulativeprobability && random <= crosspool.pool[i + 1].Cumulativeprobability) {
                //满足条件还回种群,放入第j+1个个体
                // int mark = crosspool.mark[cnt];
                population.individuals[POP_SIZE - cnt - 1] = crosspool.pool[i];
                cnt++;
                break;
            }
        }
    } 
    //对种群适应度排序
    sort(population.individuals.begin(), population.individuals.end(), compare);
    // double first = population.individuals[0].fitness > population.individuals[POP_SIZE - 1].fitness ? population.individuals[0].fitness : population.individuals[POP_SIZE - E - 1].fitness;
    cout << population.individuals[0].fitness << endl;
    // cout << "first:" << first << endl;
    // myShuffle(population.individuals.begin(), population.individuals.end());
}


/**
 * STEP6    若进化代数t=1000，则终止算法，输出适应度最大的个体作为问题的最优解。
 *          否则，令t=t+1，转入Step2
*/
int main() {
    clock_t start = clock();       
    LARGE_INTEGER seed;
    QueryPerformanceFrequency(&seed);
    QueryPerformanceCounter(&seed);
    srand(seed.QuadPart);   //初始化一个以微秒为单位的时间种子
    Population population;
    CrossPool crosspool; 
    /*STEP 1*/ populationInit(population);
    /*STEP 2*/ calculatePopulation(population);
    for(int i = 0; i < MAX_GEN; i++) {
        cout << "第" << i << "轮迭代:";
        // start = clock();
        // /*STEP 3*/ rouletteWheelSelection(population, crosspool);
        /*STEP 3-1*/ elitistArchive(population, crosspool);
        // clock_t end = clock();
        // cout<<"time3 = "<< double(end-start) <<"ms"<<endl;  //输出时间（单位：ms）
        // start = clock();
        // /*STEP 4*/ CrossAndMutate(population, crosspool);
        /*STEP 4-1*/ elitistArchiveCross(population, crosspool);
        // end = clock();
        // cout<<"time4 = "<< double(end-start) <<"ms"<<endl;  //输出时间（单位：ms）
        // start = clock();
        // /*STEP 5*/ calculateAndIterative(population, crosspool);
        /*STEP 5-1*/ elitistArchiveIterative(population, crosspool);
        // end = clock();
        // cout<<"time5 = "<< double(end-start) <<"ms"<<endl;  //输出时间（单位：ms）
        crosspool.pool.clear();
        crosspool.mark.clear();
        crosspool.pool_fitness = 0;
        population.pop_fitness = 0;
    }
    clock_t end = clock();
    cout<<"time = "<< double(end-start) <<"ms"<<endl;  //输出时间（单位：ms）
 

    system("pause");
    return 0;
}




/**
 * 注：
 * 1.在两点交叉方面，使用了random_shuffle模拟交叉
 * 2.变异方面，对两个个体分别进行变异判断
 * 3.小数的位数保留
 * 4.resize()还是reverse()呢
*/