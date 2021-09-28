
#include "iostream"
#include "vector"
#include "map"
#include "unordered_map"
using namespace std;

// int missingNumber(int nums[],int n) {
//     // int n = nums.length;
//     int res = 0;
//     // 先和新补的索引异或一下
//     res ^= n;
//     // 和其他的元素、索引做异或
//     for (int i = 0; i < n; i++)
//         res ^= i ^ nums[i];
//     return res;
// }

// int missingNumber(int nums[],int n) {
//     // int n = nums.length;
//     // 公式：(首项 + 末项) * 项数 / 2
//     int expect = (0 + n) * (n + 1) / 2; // 整数会溢出

//     int sum = 0;
//     for (int i = 0; i < n; i++)
//         sum += nums[i];
//     return expect - sum;
// }

bool isPossible_poker(vector<int>& nums) {
    unordered_map<int, int> freq;
    unordered_map<int, vector<vector<int>>> need;

    for (int v : nums) freq[v]++;

    for (int v : nums) {
        if (freq[v] == 0) {
            continue;
        }

        if (need.count(v) && need[v].size() > 0) {
            // v 可以接到之前的某个序列后面
            freq[v]--;
            // 随便取一个需要 v 的子序列
            vector<int> seq = need[v].back();
            need[v].pop_back();
            // 把 v 接到这个子序列后面
            seq.push_back(v);
            // 这个子序列的需求变成了 v + 1
            need[v + 1].push_back(seq);

        } else if (freq[v] > 0 && freq[v + 1] > 0 && freq[v + 2] > 0) {
            // 可以将 v 作为开头
            freq[v]--;
            freq[v + 1]--;
            freq[v + 2]--;
            // 新建一个长度为 3 的子序列 [v,v + 1,v + 2]
            vector<int> seq{v, v + 1, v + 2};
            // 对 v + 3 的需求加一
            need[v + 3].push_back(seq);

        } else {
            return false;
        }
    }

    // 打印切分出的所有子序列
    for (auto it : need) {
        for (vector<int>& seq : it.second) {
            for (int v : seq) {
                cout << v << " ";
            }
            cout << endl;
        }
    }

    return true;
}

bool isPossible(vector<int>& nums) {

    unordered_map<int, int> freq, need;

    // 统计 nums 中元素的频率
    for (int v : nums) freq[v]++;

    for (int v : nums) {
        if (freq[v] == 0) {
            // 已经被用到其他子序列中
            continue;
        }
        // 先判断 v 是否能接到其他子序列后面
        if (need.count(v) && need[v] > 0) {
            // v 可以接到之前的某个序列后面
            freq[v]--;
            // 对 v 的需求减一
            need[v]--;
            // 对 v + 1 的需求加一
            need[v + 1]++; 
        } else if (freq[v] > 0 && freq[v + 1] > 0 && freq[v + 2] > 0) {
            // 将 v 作为开头，新建一个长度为 3 的子序列 [v,v+1,v+2]
            freq[v]--;
            freq[v + 1]--;
            freq[v + 2]--;
            // 对 v + 3 的需求加一
            need[v + 3]++;
        } else {
            // 两种情况都不符合，则无法分配
            return false;
        }
    }

    return true;
}


int missingNumber(int nums[], int n) {
    // int n = nums.length;
    int res = 0;
    // 新补的索引
    res += n - 0;// 由于数组的特性，n在累加的时候需要整合到累加中。
    // 剩下索引和元素的差加起来
    for (int i = 0; i < n; i++) 
        res += i - nums[i];
    return res;
}

int main(void)
{
    // const int len = 4;
    // int arr[len] = {0,1,3,4};
    // int loss_num = missingNumber(arr,len);

    vector<int> nums = {1,2,3,4,5,6,7,7,8,8,9,9,10,10,11};
    // 1,2,3,4,5,5,6,7
    bool bl_is = isPossible_poker(nums);

    return 0;
}