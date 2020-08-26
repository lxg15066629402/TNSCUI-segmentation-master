# 排序
# python 实现
# 选择排序：
"""
选择排序就是取第一个数取和后面的数据进行比较，然后一轮过后得到最小的数在第一个，然后在开始取第二个数
，重复之前的比较

"""

"""
def selectionSort(nums):
    for i in range(len(nums) - 1):
        minIndex = i
        for j in range(i+1, len(nums)):
            if nums[j] < nums[minIndex]:
                minIndex = j
            if i != minIndex:
                nums[i], nums[minIndex] = nums[minIndex], nums[i]

    return nums


if __name__ == "__main__":
    nums = [1, 34, 2, 23, 23, 53, 24, 67, 33, 19, 12]
    sorted_nums = selectionSort(nums)
    print(sorted_nums)
"""


"""
动态规划问题 Dynamic Programming
可以大大的降低算法的时间复杂度
动态规划的一个常见的应用在满足最优性原理的优化问题,所谓的最优性原理指的是问题的一个最优解总是包含子问题的最优解，
但不能说明所有的子问题的最优解都最终解做贡献。
动态问题，子问题重叠
"""
# 斐波那契数列
"""
int fib(int n)
{
    if (n<=2)
        return 1;
    else:
        return fib(n-1)+fib(n-2)
}
"""
# python 实现 0-1 背包问题


def bag(n, c, w, v):
    res = [[-1 for j in range(c+1)] for i in range(n+1)]
    for j in range(c+1):
        res[0][j] = 0
    for i in range(1, n+1):
        for j in range(1, c+1):
            res[i][j] = res[i-1][j]
            if j>=w[i-1] and res[i][j] < res[i-1][j-w[i-1]] + v[i-1]:
                res[i][j] = res[i-1][j-w[i-1]] + v[i-1]
    return res


def show(n,c, w, res):
    print("最大的价值为：", res[n][c])
    x = [False for i in range(n)]
    j = c
    for i in range(1, n+1):
        if res[i][j] > res[i-1][j]:
            x[i-1] = True
            j -= w[i-1]
        print("选择物品为：")
        for i in range(n):
            if x[i]:
                print('第', i, '个', end='')
        print(" ")


if __name__ == "__main__":
    n = 5
    c = 10
    w = [2,2,6,5,4]
    v = [6,3,5,4,6]
    res = bag[n,c,w,v]
    show[n,c,w,res]