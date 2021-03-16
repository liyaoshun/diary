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

struct ListNode
{
    int val;
    ListNode * next;
    ListNode():val(0),next(nullptr){};
    ListNode(int x):val(x),next(nullptr){};
    ListNode(int x, ListNode * next):val(x),next(next){};
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


#pragma region  遍历最大二叉树

//前序遍历
void preorder_call(TreeNode * root)
{
    if (nullptr == root)
    {
        return;
    }
    cout<<root->val<<endl;
    preorder_call(root->left);
    preorder_call(root->right);
}
//中序遍历
void midorder_call(TreeNode * root)
{
    if (nullptr == root)
    {
        return;
    }
    cout<<root->val<<endl;
    midorder_call(root->left);
    midorder_call(root->right);
}
//后序遍历
void postorder_call(TreeNode * root)
{
    if (nullptr == root)
    {
        return;
    }
    cout<<root->val<<endl;
    postorder_call(root->left);
    postorder_call(root->right);
}
#pragma endregion


#pragma region  根据遍历结果来重构二叉树

// 通过前序、中序结果重构二叉树
// 前序遍历 preorder = [3,9,20,15,7]
// 中序遍历 inorder = [9,3,15,20,7]
TreeNode * engine_rebuild_btree_pm(const int preorder[], const int inorder[],int lo, int hi, int &pre_index)
{
    if (lo>hi)
    {
        pre_index -= 1;
        return nullptr;
    }
    
    int root_value = preorder[pre_index];
    
    TreeNode * root = new TreeNode(root_value);
    //根据root_value将中序（inorder）数组分为左右子树。
    int index = -1;//index表示在中序中root节点的数组下标
    for(int i = lo; i <= hi; i++)
    {
        if(root_value == inorder[i])
        {
            index = i;
            break;
        }
        else{continue;}
    }

    if (index < 0)
    {
        root->left = nullptr;
        root->right = nullptr;
        return root;
    }

    pre_index+=1;
    int l_lo = lo;
    int l_hi = index - 1;
    int r_lo = index + 1;
    int r_hi = hi;

    root->left  = engine_rebuild_btree_pm(preorder, inorder, l_lo, l_hi, pre_index);
    pre_index+=1;
    root->right = engine_rebuild_btree_pm(preorder, inorder, r_lo, r_hi, pre_index);
    return root;

}

TreeNode * rebuild_btree_from_pre_inorder(const int preorder[], const int inorder[], int len = 5)
{
    if (nullptr == preorder || nullptr == inorder )
    {
        return nullptr;
    }
    int lo = 0;int hi = len - 1;int pre_index = 0;
    return engine_rebuild_btree_pm(preorder, inorder, lo, hi, pre_index);
    
}
// 通过中序、后续结果重构二叉树
// 中序遍历 inorder = [9,3,15,20,7]
// 后序遍历 postorder = [9,15,7,20,3]
// 实现和前、中序结果重构差不多，只是需要翻转
TreeNode * engine_rebuild_btree_ip(const int postorder[], const int inorder[], int lo, int hi, int &pre_index)
{
    if (lo>hi)
    {
        pre_index += 1;
        return nullptr;
    }
    int root_value = postorder[pre_index];
    TreeNode * root = new TreeNode(root_value);
    int index = -1;
    for(int i =lo; i <= hi; i++)
    {
        if (root_value == inorder[i])
        {
            index = i;
            break;
        }else{continue;}
    }

    if (index < 0)
    {
        root->right = nullptr;
        root->left = nullptr;
        return root;
    }
    
    int l_lo = lo;int l_hi = index - 1;
    int r_lo = index + 1;int r_hi = hi;
    pre_index -= 1;
    root->right = engine_rebuild_btree_ip(postorder, inorder, r_lo, r_hi, pre_index);
    pre_index -= 1;
    root->left = engine_rebuild_btree_ip(postorder, inorder, l_lo, l_hi, pre_index);
    return root;
    
}

TreeNode * rebuild_btree_from_in_postorder(const int postorder[],const int inorder[], int len = 5)
{
    if (nullptr == postorder || nullptr == inorder )
    {
        return nullptr;
    }
    int lo = 0;int hi = len - 1;int pre_index = hi;
    return engine_rebuild_btree_ip(postorder, inorder, lo, hi, pre_index);
    
}

#pragma endregion


#pragma region  链表反转

ListNode * reverse_List(ListNode * head)
{
    if (nullptr == head->next || nullptr == head)
    {
        return head;
    }
    ListNode * last = reverse_List(head->next);
    head->next->next = head;
    head->next = nullptr;
    return last;
}

//生成单向链表
ListNode * create_List(ListNode * head, int arr [], int len, int & index)
{
    index += 1;
    if (index >= len)
    {
        head->next = nullptr;
        return nullptr;
    }
    ListNode * mid_Node = new ListNode(index);
    head->next = mid_Node;
    create_List(mid_Node, arr, len, index);
}

ListNode * create_list_engine(int arr [], int len, int & index)
{
    ListNode * head = new ListNode(index);
    create_List(head, arr, len, index);
    return head;
}

#pragma endregion



int main(void)
{
    // int a[6] = {3,2,1,6,0,5};
    // TreeNode * T = constructMaxinumBinaryTree(6, a);
    // preorder_call(T);

    // int preorder[5] = {3,9,20,15,7};
    // int inorder[5] = {9,3,15,20,7};
    // int postorder[5] = {9,15,7,20,3};
    // TreeNode * T0 = rebuild_btree_from_pre_inorder(preorder, inorder, 5);
    // TreeNode * T1 = rebuild_btree_from_in_postorder(postorder, inorder, 5);
    int len = 6;
    int arr[6] = {0,1,2,3,4,5};
    int index = 0;
    ListNode * LNode  = create_list_engine(arr, len, index);
    ListNode * LTNode = reverse_List(LNode);
    return 0;
}
