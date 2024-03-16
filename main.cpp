#include <iostream>
#include <chrono>

using namespace std;

int main()
{
    //1.矩阵运算问题
    cout<<"---Martix Multiply---"<<endl;
    int arr[500][500];
    int vec[500];
    int ans1[500] = {0};
    int ans2[500] = {0};
    for(int i=0;i<500;i++){
        for(int j=0;j<500;j++){
            arr[i][j] = (i + 1) * (j + 1);
        }
    }
    for(int i=0;i<500;i++){
        vec[i] = (i + 1);
    }
    //第一种算法
    auto start = chrono::steady_clock::now();
    for(int i=0;i<500;i++){
        int sum = 0;
        for(int j = 0; j < 500; j += 4) {
            sum += arr[j][i] * vec[j];
            sum += arr[j+1][i] * vec[j+1];
            sum += arr[j+2][i] * vec[j+2];
            sum += arr[j+3][i] * vec[j+3];
        }
        ans1[i] = sum;
    }
    auto end = chrono::steady_clock::now();
    auto duration1 = chrono::duration_cast<chrono::nanoseconds>(end - start);
    cout << "Algorithm 1 run time: " << duration1.count() << " ns" << endl;

    //第二种算法
    start = chrono::steady_clock::now();
    for(int i=0;i<500;i++){
        for(int j = 0; j < 500; j += 4) {
            ans2[j] += vec[i] * arr[i][j];
            ans2[j+1] += vec[i] * arr[i][j+1];
            ans2[j+2] += vec[i] * arr[i][j+2];
            ans2[j+3] += vec[i] * arr[i][j+3];
        }
    }
    end = chrono::steady_clock::now();
    duration1 = chrono::duration_cast<chrono::nanoseconds>(end - start);
    cout << "Algorithm 2 run time: " << duration1.count() << " ns" << endl;

    cout<<endl;

    //2.加和问题
    cout<<"---Sum Problem---"<<endl;
    int arr_a[500];
    for (int i = 0; i < 500; i += 4) {
        arr_a[i] = (i+1) * 17;
    }
    int sum1=0;
    int sum2=0;
    //普通算法
    start = chrono::steady_clock::now();
    for(int i=0;i<500;i++){
        sum1 += arr_a[i] + arr_a[i + 1] + arr_a[i + 2] + arr_a[i + 3];
    }
    end = chrono::steady_clock::now();
    duration1 = chrono::duration_cast<chrono::nanoseconds>(end - start);
    cout << "Algorithm 1 run time: " << duration1.count() << " ns" << endl;
    //超标量优化算法
    int process[10]={0};
    start = chrono::steady_clock::now();
    for(int i=0;i<50;i++){
        process[0]+=arr_a[i*10];
        process[1]+=arr_a[i*10+1];
        process[2]+=arr_a[i*10+2];
        process[3]+=arr_a[i*10+3];
        process[4]+=arr_a[i*10+4];
        process[5]+=arr_a[i*10+5];
        process[6]+=arr_a[i*10+6];
        process[7]+=arr_a[i*10+7];
        process[8]+=arr_a[i*10+8];
        process[9]+=arr_a[i*10+9];
    }
    sum2=process[0]+process[1]+process[2]+process[3]+process[4]+process[5]+process[6]+process[7]+process[8]+process[9];
    end = chrono::steady_clock::now();
    duration1 = chrono::duration_cast<chrono::nanoseconds>(end - start);
    cout << "Algorithm 2 run time: " << duration1.count() << " ns" << endl;
    system("pause");
    return 0;
}
