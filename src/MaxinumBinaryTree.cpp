#include "iostream"
using namespace std;

//最大二叉树 
//给定一个不含重复元素的整数数组。
//以此数组构建最大二叉树定义如下:
//1.二叉树的根是数组中最大的元素。 2.左子树是通过数组中最大值左边部分构建的最大二叉树。3.右子树是通过数组中最大值右边部分构建的最大二叉树.
//要求： 通过给定的数组构建最大二叉树，然后输出根节点.
//TreeNode constructMaximumBinaryTree(int[] nums);

//step 1. 弄明白根节点需要做的是什么，对于构造二叉树的话，根节点需要做的就是将自己构建出来。
//step 2. 遍历数组找到最大元素。
struct TreeNode{
    int val;
    TreeNode * left;
    TreeNode * right;
    TreeNode():val(0),left(nullptr),right(nullptr){};
    TreeNode(int x):val(x),left(nullptr),right(nullptr){};
    TreeNode(int x, TreeNode * left, TreeNode * right):val(x),left(left),right(right){};
};

#pragma region  最大二叉树构建实现代码
TreeNode * buildRoot(int lo, int hi, int arr[])
{
    if(lo > hi)
    {
        return nullptr;
    }
    if (lo == hi)
    {
        TreeNode * root = new TreeNode(arr[lo]);
        return root;
    }
    int maxvalue = -1000;
    int index = -1;
    for(int i = lo;i<=hi;++i)
    {
        int tmp = arr[i];
        if(maxvalue<tmp)
        {
            maxvalue = tmp;
            index = i;
        }else{continue;}
    }

    TreeNode * root = new TreeNode(maxvalue);
    // root->val = maxvalue;
    root->left = buildRoot(lo, index-1, arr);
    root->right = buildRoot(index+1, hi, arr);

    return root;
}

TreeNode *constructMaxinumBinaryTree(int len, int arr[])
{
    int lo = 0;
    int hi = len - 1;
    return buildRoot(lo, hi, arr);
}

#pragma endregion

int main(void)
{
    int a[6] = {3,2,1,6,0,5};
    TreeNode * T = constructMaxinumBinaryTree(6, a);
    return 0;
}