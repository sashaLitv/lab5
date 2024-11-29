#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>
#include <mpi.h>

#define SIZE 9
#define COUNT_THREADS 8

using std::vector;
using std::cout;
using std::endl;
using std::setw;

typedef vector<vector<double>> matrixD;
typedef vector<vector<int>> matrixI;

matrixD MPP;
vector<double> arrArith;
vector<double> arrGeom;
vector<double> arith;
vector<double> geom;
vector<vector<int>> top;
vector<vector<int>> topList;
matrixD matrix_n;
vector<int> id;
int Count;
int CountAll;

void do_main_task(int, int);
void initializeTree(int x);
void computeTree(int x);
void showArr(vector<double> arr);
void getArr(matrixD arr);
void clearArr();
bool updateCombination(int x, int y);
void showArithGeom(matrixD arrNow);
void createMatrix();
void showMatrix();

int main(int argc, char* argv[]) {
    bool text = true;
    double start_time, end_time;

    MPP.resize(SIZE, vector<double>(SIZE));
    arrGeom.resize(SIZE);
    arrArith.resize(SIZE);
    geom.resize(SIZE);
    arith.resize(SIZE);
    top.resize(2, vector<int>(0));
    topList.resize(2, vector<int>(0));
    matrix_n.resize(SIZE, vector<double>(SIZE));
    id.resize(SIZE);
    Count = 0;
    CountAll = 0;

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    createMatrix();
    if (rank == 0 && text) {
        showMatrix();
    }

    start_time = MPI_Wtime();
    do_main_task(rank, size);
    end_time = MPI_Wtime();
    double local_execution_time = end_time - start_time;

    vector<double> execution_times(size);
    MPI_Gather(&local_execution_time, 1, MPI_DOUBLE, execution_times.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double global_execution_time = 0.0;
    MPI_Reduce(&local_execution_time, &global_execution_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "Number of threads (MPI processes): " << size << endl;
        for (int i = 0; i < size; ++i) {
            cout << "Execution time for process " << i << ": " << execution_times[i] << " seconds" << endl;
        }
        cout << "Total execution time (max time among processes): " << global_execution_time << " seconds" << endl;
    }

    MPI_Finalize();
    return 0;
}

void createMatrix() {
   for (int i = 0; i < SIZE; i++) {
       MPP[i][i] = 1;
   }
   for (int i = 0; i < SIZE; i++) {
       for (int j = i + 1; j < SIZE; j++) {
           int temp = rand() % 2;
           if (temp) {
               MPP[i][j] = 1 / (double(rand() % 9 + 1));
           } else {
               MPP[i][j] = rand() % 9 + 1;
           }
           MPP[j][i] = 1 / MPP[i][j];

           top[0].push_back(i);
           top[1].push_back(j);
       }
   }
}

void showMatrix() {
   cout << "MPP: " << endl;
   for (int i = 0; i < SIZE; i++) {
       for (int j = 0; j < SIZE; j++) {
           cout << std::fixed << std::setprecision(2) << setw(12) << MPP[i][j];
       }
       cout << endl;
   }
   cout << endl;
}

void do_main_task(int rank, int size) {
    int x = top[0].size();
    int y = SIZE - 1;
    id.resize(x);
    for (int i = 0; i < x; i++) id[i] = i + 1;
    initializeTree(y);
    clearArr();

    // Розподіл ітерацій між процесами
    long localCountAll = 0;
    long globalCountAll = 0;

    for (long i = rank; i < pow(SIZE, SIZE - 2); i += size) { 
        updateCombination(x, y);
        initializeTree(y);
        clearArr();
        localCountAll++;
    }

    MPI_Reduce(&localCountAll, &globalCountAll, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) { // Головний процес виводить результат
        cout << "Size of matrix: " << SIZE << endl;
        cout << "Number of All trees Keili: " << pow(SIZE, SIZE - 2) << endl;
        cout << "Number of ALL trees: " << globalCountAll << endl;
    }

    if (rank == 0) { // Вивід результатів арифметичної та геометричної прогресії
        showArithGeom(MPP);
    }
}


void initializeTree(int x) {
   topList[0].resize(x);
   topList[1].resize(x);

   for (int i = 0; i < x; i++) {
       topList[0][i] = top[0][id[i] - 1];
       topList[1][i] = top[1][id[i] - 1];
   }


   computeTree(x);
}

void computeTree(int x){
   for(int i = 0; i < SIZE; i++) {
       matrix_n[i][i] = 1;
   }
    
   for (int i = 0; i < SIZE; i++) {
       for (int j = i + 1; j < SIZE; j++) {
           for (int k = 0; k < x; k++) {
               if (i == topList[0][k] && j == topList[1][k]) {
                   matrix_n[i][j] = MPP[i][j];
               }
           }
       }
   }

   for (int i = 0; i < SIZE - 1; i++) {
       for (int j = i + 1; j < SIZE; j++) {
           for (int k = 0; k < SIZE; k++) {
               if (matrix_n[i][j] == 0 && MPP[i][k] != 0 && MPP[j][k] != 0) {
                   matrix_n[i][j] = MPP[i][k] * MPP[j][k];
               }
           }
       }
   }

   for (int i = 0; i < SIZE; i++) {
       for (int j = i; j < SIZE; j++) {
           matrix_n[j][i] = 1 / matrix_n[i][j];
       }
   }
}

void clearArr() {
   for (int i = 0; i < SIZE; i++) {
       for (int j = 0; j < SIZE; j++) {
           matrix_n[i][j] = 0;
       }
   }

   for (int i = 0; i < SIZE; i++) {
       arrGeom[i] = 0;
       arrArith[i] = 0;
   }
}

bool updateCombination(int x, int y) {
   for(int i = y - 1; i >= 0; --i) {
       if (id[i] < x - (y - 1 - i)) {
           ++id[i];
           for (int j = i + 1; j < y; ++j)
               id[j] = id[j - 1] + 1;
           return true;
       }
   }
   return false;
}

void getArr(matrixD arr) {
    double start_arith, end_arith, start_geom, end_geom;
    double sumVarith = 0;
    double sumVgeom = 0;
    double productGeom = 1;
    
    start_arith = MPI_Wtime();
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            arith[i] += arr[i][j];
        }
        arith[i] = arith[i] / SIZE;
        sumVarith += arith[i];
    }
    end_arith = MPI_Wtime();
    
    start_geom = MPI_Wtime();
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            productGeom *= arr[i][j];
        }
        geom[i] = pow(productGeom, (1.0 / SIZE));
        productGeom = 1;
        sumVgeom += geom[i];
    }
    end_geom = MPI_Wtime();

    for (int i = 0; i < SIZE; i++) {
        arrArith[i] = arith[i] / sumVarith;
        arrGeom[i] = geom[i] / sumVgeom;
    }
}
void showArithGeom(matrixD arrNow) {
   double sumGeom = 0;
   double sumArith = 0;

   getArr(arrNow);

   cout << "Arithmetic Mean: ";
   for (int i = 0; i < SIZE; i++) {
       sumArith += arrArith[i];
       cout << std::fixed << std::setprecision(2) << setw(10) << arrArith[i];
   }
   cout << " => sum: " << sumArith << endl;

   cout << "Geometric Mean: ";
   for (int i = 0; i < SIZE; i++) {
       sumGeom += arrGeom[i];
       cout << std::fixed << std::setprecision(2) << setw(10) << arrGeom[i];
   }
   cout << " => sum: " << sumGeom << endl;
}

