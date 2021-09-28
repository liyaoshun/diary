#   <div align = center>**C++/编程等相关知识** </div>

## **C++中基类的析构函数为什么要用virtual虚析构函数**

```
大家知道，析构函数是为了在对象不被使用之后释放它的资源，虚函数是为了实现多态。那么把析构函数声明为vitual有什么作用呢？ 直接的讲，C++中基类采用virtual虚析构函数是为了防止内存泄漏。具体地说，如果派生类中申请了内存空间，并在其析构函数中对这些内存空间进行释放。假设基类中采用的是非虚析构函数，当删除基类指针指向的派生类对象时就不会触发动态绑定，因而只会调用基类的析构函数，而不会调用派生类的析构函数。那么在这种情况下，派生类中申请的空间就得不到释放从而产生内存泄漏。所以，为了防止这种情况的发生，C++中基类的析构函数应采用virtual虚析构函数。
————————————————
```
```
#include <iostream>
using namespace std;
 
class Base
{
public:
    Base() {}; //Base的构造函数
    ~Base() //Base的析构函数
    {
        cout << "Output from the destructor of class Base!" << endl;
    };
    virtual void DoSomething()
    {
        cout << "Do something in class Base!" << endl;
    };
};
 
class Derived : public Base
{
public:
    Derived() {}; //Derived的构造函数
    ~Derived() //Derived的析构函数
    {
        cout << "Output from the destructor of class Derived!" << endl;
    };
    void DoSomething()
    {
        cout << "Do something in class Derived!" << endl;
    };
};
 
int main()
{
    Derived *pTest1 = new Derived(); //Derived类的指针
    pTest1->DoSomething();
    delete pTest1;
 
    cout << endl;
 
    Base *pTest2 = new Derived(); //Base类的指针
    pTest2->DoSomething();
    delete pTest2;
 
    return 0;
}

可以正常释放pTest1的资源，而没有正常释放pTest2的资源，因为从结果看Derived类的析构函数并没有被调用。通常情况下类的析构函数里面都是释放内存资源，而析构函数不被调用的话就会造成内存泄漏。原因是指针pTest2是Base类型的指针，释放pTest2时只进行Base类的析构函数
————————————————

在代码~Base析构函数前面加上virtual关键字后，此时释放指针pTest2时，由于Base的析构函数是virtual的，就会先找到并执行Derived类的析构函数，然后再执行Base类的析构函数，资源正常释放，避免了内存泄漏。
因此，只有当一个类被用来作为基类的时候，才会把析构函数写成虚函数。

```

---
## **二叉树相关知识**
### **最大二叉树**

**实现代码:** [code](src/MaxinumBinaryTree.cpp)

**参考地址:** [reference](https://github.com/labuladong/fucking-algorithm)

---
## **杂项**
```
关于在for循环里面使用i++还是使用++i的问题。
对于i++都知道是当前for里面先使用i的值计算，在for计算完成后的时候i的值才加1用于判断，此时就需要额外的申请空间来保存+1后的变量。而++i的含义为当前for里面直接使用，但是不需要额外的申请空间。在旧版本的编译器里面使用++i会有效率的提升，但是现在的新版本的编译器是没有这个问题了。
```
---
## **C++ 容器使用**
###  **线性容器**
```
1. 为什么要引入 std::array 而不是直接使用 std::vector?
    与 std::vector 不同,std::array 对象的大小是固定的,如果容器大小是固定的,那么可以优先考虑使用 std::array 容器。另外由于 std::vector 是自动扩容的,当存入大量的数据后,并且对容器进行了删除操作,容器并不会自动归还被删除元素相应的内存,这时候就需要手动运行 shrink_to_fit() 释放这部分内存。

2. 已经有了传统数组,为什么要用 std::array?
    使用 std::array 能够让代码变得更加 ‘‘现代化’’,而且封装了一些操作函数,比如获取数组大小以及检查是否非空,同时还能够友好的使用标准库中的容器算法,比如 std::sort。
```
