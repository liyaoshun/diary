#include <vector>
#include <map>
#include <cstring>
#include <unordered_map>

using namespace  std;

struct TreeNode{
    int val;
    TreeNode * left;
    TreeNode * right;
    TreeNode():val(0),left(nullptr),right(nullptr){};
    TreeNode(int x):val(x),left(nullptr),right(nullptr){};
    TreeNode(int x, TreeNode * left, TreeNode * right):val(x),left(left),right(right){};
};

class Solution {
private:
    // 结果的存储
    vector<TreeNode*> res;
    // 子树的编号，从1开始，用0表示是无效，后续递增来保证唯一
    int index = 1;
    // 子树 val,left,right的字符串 到唯一编号的映射
    unordered_map<string, int> str2index;
    // 唯一编号 到 数量的映射
    // unordered_map<int, int> index2cnt;
    // 这里假设是优先编号，如5000已满足测试需求来提速
    int index2cnt[5000];

    // 返回值为 当前节点对应的 唯一编号
    int dfs(TreeNode* curr)
    {
        if (curr != nullptr)
        {
            string currStr = to_string(curr->val) + "," + to_string(dfs(curr->left)) + "," + to_string(dfs(curr->right));
            if (str2index.find(currStr) == str2index.end())
            {
                str2index[currStr] = index;
                ++index;
            }
            //计算当前节点在映射表中的编号，并将编号加一（初始都是0）
            int index_find = str2index[currStr];
            ++index2cnt[index_find];
            
            // 首次发现重复，增加结果里
            if (2 == index2cnt[index_find])
            {
                res.push_back(curr);
            }
            return index_find;
        }
        else
        {
            return 0;
        }
    }

public:
    vector<TreeNode*> findDuplicateSubtrees(TreeNode* root) {
        dfs(root);
        memset(index2cnt, 0, sizeof(int) * 5000);
        return res;
    }
};