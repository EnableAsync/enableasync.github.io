---
title: 算法整理
date: 2022-01-04 19:21:50
tags: data structure, algorithm
image: https://dingxuewen.com/leetcode-js-leviding/leetcode.png
mathjax: true
typora-root-url: ./leetcode
---

# 注意事项

## `rust` 中 `dbg!` 超时

在 `rust` 中使用 `dbg!` 的时候，在题目判定时，可能会因为 `dbg!` 超时，提交代码的时候要去掉 `dbg!`

# 数组
## 双指针
第 27、977 题就是经典的双指针题目。

- 有序数组平方。

## 滑动窗口
注意，使用滑动窗口的时候，只用一个 for 循环代表滑动窗口的结尾，否则又会陷入两个 for 的困境。

- 长度最小的子数组

## 前缀和

```Java
for(int i = 1; i <= n; i++) {
    s[i] = s [i-1] + a[i];
}
```

## Java 多组输入示例

```java
import java.util.Scanner;

public class Main {
    public static void main (String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        int n = scanner.nextInt();
        int[] nums = new int[n + 1];
        int[] s = new int[n + 1];
        
        for (int i = 1; i <= n; i++) {
            nums[i] = scanner.nextInt();
            s[i] = s[i - 1] + nums[i];
        }
        
        while (scanner.hasNextInt()) {
            int a = scanner.nextInt();
            int b = scanner.nextInt();
            System.out.println(s[b+1] - s[a]);
        }
        scanner.close();
    }
}
```

## 模拟矩阵

1. 走完一行或者一列对 x 和 y 进行处理，使得继续走下一行而不下标越界
2. 走完一行或者一列对 x 和 y 进行处理，使得走到没有写数字的地方
3. 走完一行或者一列对边界进行处理

```java
class Solution {
    public int[][] generateMatrix(int n) {
        int [][]result = new int[n][n];
        int count = 1;
        int x = 0, y = 0;
        int xMax = n, yMax = n, xMin = 0, yMin = 0;
        while (count <= n * n) {
            // 向右
            while (y < yMax) {
                result[x][y] = count++;
                y++;
            }
            y--; // 能够向下走
            x++; // 走到没写过数字的地方
            xMin += 1; // 向右一行补充完向上少走一行

            // 向下
            while (x < xMax) {
                result[x][y] = count++;
                x++;
            }
            x--; // 能够向左走
            y--; // 走到没写过数字的地方
            yMax -= 1; // 向右少走一行

            // 向左
            while (y >= yMin) {
                result[x][y] = count++;
                y--;
            }
            y++; // 能够向右走
            x--; // 走到没写过数字的地方
            xMax -= 1; // 向下少走一行

            // 向上
            while (x >= xMin) {
                result[x][y] = count++;
                x--;
            }
            x++; // 能够向右走
            y++; // 走到没写过数字的地方
            yMin += 1; // 向左少走一行

        }
        return result;
    }
    
}
```

# 矩阵

## 矩阵置零

给定一个 **`m x n`** 的矩阵，如果一个元素为 **0** ，则将其所在行和列的所有元素都设为 **0** 。请使用 **[原地](http://baike.baidu.com/item/原地算法)** 算法。

难点在于原地算法，否则很简单，模拟就好，模拟的时候注意不要跳过本来为 0 的元素。



# 双指针

## 移动零

10^4 O(n)

给定一个数组 `nums`，编写一个函数将所有 `0` 移动到数组的末尾，同时保持非零元素的相对顺序。

**请注意** ，必须在不复制数组的情况下原地对数组进行操作。

**key：**保证 right 在 left 左边。

## 盛最多水的容器

10^5 O(n)

给定一个长度为 `n` 的整数数组 `height` 。有 `n` 条垂线，第 `i` 条线的两个端点是 `(i, 0)` 和 `(i, height[i])` 。

找出其中的两条线，使得它们与 `x` 轴共同构成的容器可以容纳最多的水。

返回容器可以储存的最大水量。

**说明：**你不能倾斜容器。

**思考：**有两个变量决定盛水量：1. 左右的距离。 2. 较低的柱子高度。

1. 左右的距离最左到最右最大，然后慢慢缩小。
2. 如果移动较高的柱子，那么盛水量不会变大，但是如果移动较低的柱子，那么盛水量有可能会变大。

## 接雨水

2 * 10^4

![img](rainwatertrap.png)

```
输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。 
```

**key：**某一处的雨水 = 左右最高柱子的最小值 - 当前处的高度

剩下的点就在求左右最高柱子处进行优化。

### 动态规划

将左右最高柱子的高度分别记为 leftMax，rightMax，并 O(n) 计算出这个数组的。

需要注意的点是 **初始值** 和 **边界**。

```java
class Solution {
    public int trap(int[] height) {
        int n = height.length;
        int[] leftMax = new int[n];
        int[] rightMax = new int[n];
        leftMax[0] = height[0];
        for (int i = 1; i < n; i++) {
            leftMax[i] = Math.max(leftMax[i - 1], height[i]);
        }
        rightMax[n - 1] = height[n - 1];
        for (int i = n - 2; i >= 0; i--) {
            rightMax[i] = Math.max(rightMax[i + 1], height[i]);
        }
        int total = 0;
        for (int i = 1; i < n; i++) {
            total += Math.min(leftMax[i], rightMax[i]) - height[i];
        }
        return total;
    }
}
```

时间复杂度：O(n)，其中 n 是数组 height 的长度。计算数组 leftMax 和 rightMax 的元素值各需要遍历数组 height 一次，计算能接的雨水总量还需要遍历一次。

空间复杂度：O(n)，其中 n 是数组 height 的长度。需要创建两个长度为 n 的数组 leftMax 和 rightMax。

### 单调栈



### 双指针

动态规划计算 leftMax 和 rightMax 的时候需要遍历一次数组，能不能直接得到 leftMax 和 rightMax 而不遍历呢？这样就可以将得到 max 的复杂度降低为 O(1) 了。

**与盛最多水的容器相同的思路，可以从两边的雨水向中间进行计算**，这样可以 O(1) 得到

```java
class Solution {
    public int trap(int[] height) {
        int n = height.length;
        int ans = 0;
        int left = 0, right = n - 1;
        int maxLeft = 0, maxRight = 0;
        while (left < right) {
            maxLeft = Math.max(maxLeft, height[left]);
            maxRight = Math.max(maxRight, height[right]);
            if (height[left] > height[right]) {
                ans += maxRight - height[right];
                right--;
            } else {
                ans += maxLeft - height[left];
                left++;
            }
        }
        return ans;
    }
}
```

# 滑动窗口

### 滑动窗口模板

```java
// 外层循环扩展右边界，内层循环扩展左边界
for (int l = 0, r = 0 ; r < n ; r++) {
	// 当前考虑的元素
	while (l <= r && check()) { // 区间[left,right]不符合题意
        // 扩展左边界
    }
    // 区间[left,right]符合题意，统计相关信息
}
```

## 无重复字符的最长子串

5 * 10^4 级别

给定一个字符串 `s` ，请你找出其中不含有重复字符的最长子串的长度。

1. 判断重复字符：Set
2. 最长字串：滑动窗口

这道题官方解答为枚举所有起始位置，之后向右延申至不含重复字符的最长子串长度，如果包含重复字符，那么左指针持续移动到不含重复字符为止。

关键在于有重复字符的时候，是左指针持续地向右移动，而不是重新开始枚举，因为这个操作，使得时间复杂度为 O(n)，那么为什么这样不会漏掉答案呢？也就是说为什么这样一定会取到最优答案呢？

因为当右边要出现重复字母的的候，这个时候左指针到右指针的子串一定是对应了右指针不移动情况下的最优子串。

所以可以枚举右指针，并让左指针不断向右。

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        Set<Character> set = new HashSet<>();
        int n = s.length(), l = 0, ans = 0;
        for (int r = 0; r < n; r++) {
            while (set.contains(s.charAt(r))) {
                set.remove(s.charAt(l));
                l++;
            }
            set.add(s.charAt(r));
            ans = Math.max(ans, r - l + 1);
        }
        return ans;
    }
}
```

## 找到字符串中所有字母异位词

给定两个字符串 `s` 和 `p`，找到 `s` 中所有 `p` 的 异位词的子串，返回这些子串的起始索引。不考虑答案输出的顺序。

3 * 10^4 级别

与上一题相同，需要做以下操作：

1. 判断异位词。
2. 异位词子串。

确定异位词需要 O(n)，n 是字符串长度。

**key：**在这道题中，异位词和原词肯定是长度相同的，所以直接用固定长度的滑动窗口滑过去，这样时间复杂度为 O(n * m)。但是有更好的做法，就是当向右滑动的时候，删掉的是左面的字母，如果右边能够补齐，那么就说明是异位词，这样就可以 O(1) 判断异位词，再加上滑动的耗时，总计 O(m)。

Arrays.equals 可以判断两个数组相等。

# 链表
链表对于有插入、交换或者删除的操作的时候，一般加一个**虚拟头节点**更好处理。

203.移除链表元素

707.设计链表

206.翻转链表

206.翻转链表

19.删除链表的倒数第 N 个结点

## K 个一组翻转链表

给你链表的头节点 `head` ，每 `k` 个节点一组进行翻转，请你返回修改后的链表。

`k` 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 `k` 的整数倍，那么请将最后剩余的节点保持原有顺序。

你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。



**Key：**关键在于写一个 reverse 函数逆转从 head 到 tail 的链表。其中 reverse 函数可以返回逆转后的最后一个节点。还有需要维护返回的节点。

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode reverseKGroup(ListNode head, int k) {
        if (k == 1) return head;
        boolean first = true;
        ListNode dummy = new ListNode(0, head), cur = head, pre = dummy, ret = null;
        int count = 0;
        while (cur != null) {
            count++;
            if (count == k) {
                if (first) {
                    ret = cur;
                    first = false;
                }
                pre = reverse(pre, head, cur);
                head = pre.next;
                cur = pre;
                count = 0;
            }
            cur = cur.next;
        }
        return ret;
    }

    // 逆转从 head 到 tail 的链表，pre 是 head 的前一个结点
    // 返回逆转后的最后一个节点，其实就是 head，这一组的最后一个节点是下一组的 pre
    private ListNode reverse(ListNode pre, ListNode head, ListNode tail) {
        // pre -> head -> cur -> tail
        ListNode cur = head.next, retTail = head;
        head.next = tail.next;
        while (cur != tail) { // tail 之前的节点全部头插法插到到 pre 之后
            ListNode next = cur.next;
            pre.next = cur;
            cur.next = head;
            head = cur;
            cur = next;
        }
        // 把 tail 也插到头部
        pre.next = tail;
        tail.next = head;
        return retTail;
    }
}
```

## 排序链表

要点在于排序 + 链表操作

题目的进阶问题要求达到 O(nlogn) 的时间复杂度和 O(1) 的空间复杂度，时间复杂度是 O(nlogn) 的排序算法包括归并排序、堆排序和快速排序（快速排序的最差时间复杂度是 O(n^2)），其中最适合链表的排序算法是归并排序。

归并排序基于分治算法。最容易想到的实现方式是自顶向下的递归实现，考虑到递归调用的栈空间，自顶向下归并排序的空间复杂度是 O(logn)。如果要达到 O(1) 的空间复杂度，则需要使用自底向上的实现方式。=

递归写法：

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode mid = findMiddle(head);
        ListNode rightHead = mid.next;
        mid.next = null; // 断开链表

        // 排序
        ListNode left = sortList(head);
        ListNode right = sortList(rightHead);
        
        // 合并有序链表
        return mergeList(left, right);
    }

    private ListNode mergeList(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(0);
        ListNode cur = dummy;
        while (l1 != null && l2 != null) {
            if (l1.val < l2.val) {
                cur.next = l1;
                l1 = l1.next;
            } else {
                cur.next = l2;
                l2 = l2.next;
            }
            cur = cur.next;
        }
        // 接上剩余链表，这里不需要用 while，因为肯定要么 l1 要么 l2 剩余，剩余部分本来就是接好的
        if (l1 != null) {
            cur.next = l1;
        }

        if (l2 != null) {
            cur.next = l2;
        }

        return dummy.next;
    }

    private ListNode findMiddle(ListNode head) {
        ListNode slow = head;
        ListNode fast = head.next; // 重要！fast 从 head.next 开始，确保 slow 指向中点或者左中点
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }
}
```

迭代写法：

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    // 1. 获取链表长度
    // 2. 设置合并的长度（step）
    // 3. 合并根据长度划分的所有链表
    // 4. step *= 2
    public ListNode sortList(ListNode head) {
        ListNode dummy = new ListNode(0, head);
        int length = listLength(head);
        for (int step = 1; step < length; step *= 2) {
            ListNode cur = dummy.next;
            ListNode newTail = dummy;
            while (cur != null) {
                // 从 cur 开始，分割出两段长为 step 的链表，头节点分别为 head1 和 head2
                ListNode head1 = cur;
                ListNode head2 = splitList(head1, step);
                // 下一轮的起点，也是为了分割开链表
                cur = splitList(head2, step); 
                // 合并两段长度为 step 的链表
                ListNode merged = mergeList(head1, head2);
                // 找到下一个要合并的链表的 head
                int len = 0;
                ListNode curr = merged;
                while(len < step * 2 && curr.next != null) {
                    len++;
                    curr = curr.next;
                }
                // 合并后的头节点插入到 newTail 后面
                newTail.next = merged;
                newTail = curr;
            }
        }
        return dummy.next;
    }

    private ListNode splitList(ListNode head, int size) {
        ListNode cur = head;
        // nextHead 的前一个节点，用于断开链表
        for (int i = 0; i < size - 1 && cur != null; i++) {
            cur = cur.next;
        }

        if (cur == null || cur.next == null) return null;

        ListNode nextHead = cur.next;
        cur.next = null; // 断开链表
        return nextHead;
    }

    private int listLength(ListNode head) {
        int length = 0;
        while (head != null) {
            head = head.next;
            length++;
        }
        return length;
    }

    // 返回合并链表的头节点
    private ListNode mergeList(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(0);
        ListNode cur = dummy;
        while (l1 != null && l2 != null) {
            if (l1.val < l2.val) {
                cur.next = l1;
                l1 = l1.next;
            } else {
                cur.next = l2;
                l2 = l2.next;
            }
            cur = cur.next;
        }
        // 接上剩余链表，这里要用 while，因为肯定要么 l1 要么 l2 剩余，剩余部分本来就是接好的
        if (l1 != null) {
            cur.next = l1;
        }

        if (l2 != null) {
            cur.next = l2;
        }

        return dummy.next;
    }
}
```

## 合并 K 个升序链表

### 朴素做法

用一个数组存储所有的 head，然后每次取出所有数组的最小值，然后插入到链表最后面。

每次取出最小值，复杂度为 O(k)，然后一共要取 k * n 次，n 是最长链表的长度，这样时间复杂度为 O(k^2 * n)，每次取出最小值的时候，有比较被浪费掉了。

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        ListNode dummy = new ListNode(0);
        ListNode cur = dummy;
        ListNode[] head = new ListNode[lists.length];
        int allLength = 500 * 10000;
        for (int i = 0; i < lists.length; i++) {
            head[i] = lists[i];
        }

        int i = 0;
        while (i < allLength) {
            int minVal = 0x3f3f3f3f;
            int minIndex = -1;
            for (int j = 0; j < lists.length; j++) {
                if (head[j] != null && minVal > head[j].val) {
                    minVal = head[j].val;
                    minIndex = j;
                }
            }
            // 所有链表都为空
            if (minIndex == -1) break;
            cur.next = head[minIndex];
            cur = cur.next;
            head[minIndex] = head[minIndex].next;
            i++;
        }
        return dummy.next;
    }
}
```

### 二分做法

![img](./6f70a6649d2192cf32af68500915d84b476aa34ec899f98766c038fc9cc54662-image.png)

复杂度计算：

![image-20250202004335203](./image-20250202004335203.png)

链表两两合并避免比较浪费，左闭右闭实现：

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        return mergeLists(lists, 0, lists.length - 1);
    }

    private ListNode mergeLists(ListNode[] lists, int l, int r) {
        if (l == r) return lists[l];
        if (l > r) return null;
        int mid = l + (r - l) / 2;
        // 左闭右闭
        return mergeTwoLists(mergeLists(lists, l, mid), mergeLists(lists, mid + 1, r));
    }

    private ListNode mergeTwoLists(ListNode a, ListNode b) {
        ListNode dummy = new ListNode(0);
        ListNode cur = dummy;
        while (a != null && b != null) {
            if (a.val <= b.val) {
                cur.next = a;
                a = a.next;
            } else {
                cur.next = b;
                b = b.next;
            }
            cur = cur.next;
        }

        if (a != null) cur.next = a;
        if (b != null) cur.next = b;

        return dummy.next;
    }
}
```

左闭右开实现：

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        return mergeLists(lists, 0, lists.length);
    }

    private ListNode mergeLists(ListNode[] lists, int l, int r) {
        if (l >= r) return null;
        if (r - l == 1) return lists[l];
        int mid = l + (r - l) / 2;
        // 左闭右开
        return mergeTwoLists(mergeLists(lists, l, mid), mergeLists(lists, mid, r));
    }

    private ListNode mergeTwoLists(ListNode a, ListNode b) {
        ListNode dummy = new ListNode(0);
        ListNode cur = dummy;
        while (a != null && b != null) {
            if (a.val <= b.val) {
                cur.next = a;
                a = a.next;
            } else {
                cur.next = b;
                b = b.next;
            }
            cur = cur.next;
        }

        if (a != null) cur.next = a;
        if (b != null) cur.next = b;

        return dummy.next;
    }
}
```

# 二叉树

## 二叉树的中序遍历

递归做法：

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    List<Integer> list = new ArrayList<>();

    public List<Integer> inorderTraversal(TreeNode root) {
        helper(root);
        return list;
    }

    private void helper(TreeNode root) {
        if (root == null) {
            return;
        }
        helper(root.left);
        list.add(root.val);
        helper(root.right);
    }
}
```

非递归做法，用一个栈模拟递归栈。

前序遍历是中左右，如果还有左子树就一直向下找。完了之后再返回从最底层逐步向上向右找。

```java
class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> list = new ArrayList<>();
        Deque<TreeNode> stack = new LinkedList<>();
        while (root != null || !stack.isEmpty()) {
            while (root != null) {
                list.add(root.val);
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            root = root.right;
        }
        return list;
    }
}
```



中序是左中右，如果还有左子树就一直向下找，直到左边最底部，然后处理节点：

```java
class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> list = new ArrayList<>();
        Deque<TreeNode> stack = new LinkedList<>();
        while (root != null || !stack.isEmpty()) {
            while (root != null) {
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            list.add(root.val);
            root = root.right;
        }
        return list;
    }
}
```

前序是先中间，再左边然后右边，而这里是先中间，再后边然后左边。那我们完全可以改造一下前序遍历，得到序列new_seq之后再reverse一下就是想要的结果了：

```java
class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> list = new ArrayList<>();
        Deque<TreeNode> stack = new LinkedList<>();
        while (root != null || !stack.isEmpty()) {
            while (root != null) {
                list.add(root.val);
                stack.push(root);
                root = root.right;
            }
            root = stack.pop();
            root = root.left;
        }
        Collections.reverse(list);
        return list;
    }
}
```

## 二叉树的最大深度

要点在于如何记录深度，递归的时候可以通过传参解决：

```java
class Solution {
    int maxDepth = 0;

    public int maxDepth(TreeNode root) {
        helper(root, 0);
        return maxDepth;
    }

    private void helper(TreeNode root, int depth) {
        if (root == null) {
            maxDepth = Math.max(maxDepth, depth);
            return;
        }
        helper(root.left, depth + 1);
        helper(root.right, depth + 1);
    }
}
```

```java
class Solution {
    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        } else {
            int leftHeight = maxDepth(root.left);
            int rightHeight = maxDepth(root.right);
            return Math.max(leftHeight, rightHeight) + 1;
        }
    }
}
```

## 反转二叉树

```java
class Solution {
    public TreeNode invertTree(TreeNode root) {
        helper(root);
        return root;
    }

    private TreeNode helper(TreeNode root) {
        if (root == null) return null;
        TreeNode left = helper(root.right);
        TreeNode right = helper(root.left);
        root.left = left;
        root.right = right;
        return root;
    }
}
```

## 对称二叉树

这里注意条件是 `left.val == right.val && helper(left.left, right.right) && helper(left.right, right.left)` ，因为对称是中心轴对称，而不是左右相等。

```java 
class Solution {
    List<Integer> list = new ArrayList<>();
    public boolean isSymmetric(TreeNode root) {
        return helper(root.left, root.right);
    }

    private boolean helper(TreeNode left, TreeNode right) {
        if (left == null && right == null) {
            return true;
        }
        if (left == null || right == null) {
            return false;
        }
        return left.val == right.val
            && helper(left.left, right.right)
            && helper(left.right, right.left);
    }
}
```

## 二叉树的直径

二叉树的直径 = 最深左子树深度 + 最深右子树深度

```java
class Solution {
    int ans = 0;
    public int diameterOfBinaryTree(TreeNode root) {
        helper(root);
        return ans;
    }

    private int helper(TreeNode root) {
        if (root == null) return 0;
        int leftMax = helper(root.left);
        int rightMax = helper(root.right);
        ans = Math.max(ans, leftMax + rightMax);
        return Math.max(leftMax, rightMax) + 1;
    }
}
```

## 二叉树层序遍历

我的做法：先序遍历（root，左，右）的时候记住 depth 存到 map 中，然后最后把 map 中的数组组合起来：

```java
class Solution {
    List<List<Integer>> ans = new ArrayList<>();
    Map<Integer, List<Integer>> map = new HashMap<>();

    public List<List<Integer>> levelOrder(TreeNode root) {
        int depth = helper(root, 0);
        for (int i = 0; i < depth; i++) {
            ans.add(map.get(i));
        }
        return ans;
    }

    private int helper(TreeNode root, int depth) {
        if (root == null) return depth;
        List<Integer> arr = map.getOrDefault(depth, new ArrayList<Integer>());
        arr.add(root.val);
        map.put(depth, arr);
        int leftMax = helper(root.left, depth + 1);
        int rightMax = helper(root.right, depth + 1);
        return Math.max(leftMax, rightMax);
    }
}
```

bfs 做法：

- root 入队列
- 队列不为空的时候
  - 求当前队列长度 $s_i$
  - 取 $s_i$ 个元素进行拓展，进入下一次迭代

它和普通广度优先搜索的区别在于，普通广度优先搜索每次只取一个元素拓展，而这里每次取 $s_i$ 个元素。在上述过程中的第 $i$ 次迭代得到了二叉树第 $i$ 层的 $s_i$ 个元素。（说白了就是每次迭代的时候把下一层级的所有元素都加到队列里面，这样每一次迭代整个队列元素的时候就是一个层级的所有元素）

```java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> ans = new ArrayList<>();
        if (root == null) return ans;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            List<Integer> list = new ArrayList<>();
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode front = queue.poll();
                if (front.left != null) queue.offer(front.left);
                if (front.right != null) queue.offer(front.right);
                list.add(front.val);
            }
            ans.add(list);
        }
        return ans;
    }
}
```

## 将有序数组转换为二叉搜索树

递归建树：

- `1 <= nums.length <= 10^4`
- `-10^4 <= nums[i] <= 10^4`
- `nums` 按 **严格递增** 顺序排列



![image-20250205185730436](./image-20250205185730436.png)

```java
class Solution {
    public TreeNode sortedArrayToBST(int[] nums) {
        return helper(nums, 0, nums.length);
    }

    private TreeNode helper(int[] nums, int left, int right) {
        if (left >= right) return null;
        int mid = left + (right - left) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = helper(nums, left, mid);
        root.right = helper(nums, mid + 1, right);
        return root;
    }
}
```

## 验证二叉搜索树

给你一个二叉树的根节点 `root` ，判断其是否是一个有效的二叉搜索树。

**有效** 二叉搜索树定义如下：

- 节点的左子树只包含 **小于** 当前节点的数。
- 节点的右子树只包含 **大于** 当前节点的数。
- 所有左子树和右子树自身必须也是二叉搜索树。

这里有一个问题是需要判断所有的子节点都大于或者都小于根节点。

**二叉搜索树的中序遍历是递增的。**所以可以中序遍历，之后每个元素都小于前一个元素，则是二叉搜索树。

递归写法：

```java
class Solution {
    List<Integer> list = new ArrayList<>();

    public boolean isValidBST(TreeNode root) {
        helper(root);
        for (int i = 0; i < list.size() - 1; i++) {
            if (list.get(i) >= list.get(i + 1)) {
                return false;
            }
        }
        return true;
    }

    private void helper(TreeNode root) {
        if (root == null) return;
        helper(root.left);
        list.add(root.val);
        helper(root.right);
    }
}
```

递归写法2：

```java
class Solution {
    long left = Long.MIN_VALUE;
    public boolean isValidBST(TreeNode root) {
        return helper(root);
    }

    private boolean helper(TreeNode root) {
        if (root == null) return true;
        boolean l = helper(root.left);
        boolean tmp = left < root.val;
        left = root.val;
        boolean r = helper(root.right);
        return l && r && tmp;
    }
}
```

递归写法3：

```java
class Solution {
    public boolean isValidBST(TreeNode root) {
        return helper(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }

    private boolean helper(TreeNode root, long lower, long upper) {
        if (root == null) return true;
        if (root.val <= lower || root.val >= upper) return false;
        return helper(root.left, lower, root.val) && helper(root.right, root.val, upper);
    }
}
```



迭代写法：

```java
class Solution {
    public boolean isValidBST(TreeNode root) {
        return helper(root);
    }

    private boolean helper(TreeNode root) {
        Deque<TreeNode> stack = new LinkedList<TreeNode>();
        long left = Long.MIN_VALUE;
        while (root != null || !stack.isEmpty()) {
            while (root != null) {
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            if (left >= root.val) return false;
            left = root.val;
            root = root.right;
        }
        return true;
        
    }
}
```

## 二叉搜索树中第 K 小的元素

和验证二叉搜索树一样，在最外层存储一下状态。
```java
class Solution {
    int count = 0, target; // 存储状态
    public int kthSmallest(TreeNode root, int k) {
        target = k;
        return helper(root);
    }

    private int helper(TreeNode root) {
        if (root == null) return -1; 
        int left = helper(root.left);
        if (left != -1) return left;
        count++;
        if (target == count) {
            return root.val;
        }
        int right = helper(root.right);
        if (right != -1) return right;
        return -1;
    }
}
```

## 二叉树的右视图

给定一个二叉树的 **根节点** `root`，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。

层序遍历中最右面的那个元素就是答案，放进去就好了。

```java
class Solution {
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> ans = new ArrayList<>();
        if (root == null) return ans;
        Deque<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
                if (i == size - 1) {
                    ans.add(node.val);
                }
            }
        }
        return ans;
    }
}
```

## 二叉树展开为链表

给你二叉树的根结点 `root` ，请你将它展开为一个单链表：

- 展开后的单链表应该同样使用 `TreeNode` ，其中 `right` 子指针指向链表中下一个结点，而左子指针始终为 `null` 。
- 展开后的单链表应该与二叉树 [**先序遍历**](https://baike.baidu.com/item/先序遍历/6442839?fr=aladdin) 顺序相同。

### 递归

先序遍历二叉树，然后记录 last，然后修改 last 的 left 和 right，但是这样会栈溢出。

原因是递归遍历 root.left 的时候，root.right 被改成了 root.left 导致死循环，保存一下状态就好了。

```java
class Solution {
    private TreeNode last;
    public void flatten(TreeNode root) {
        last = new TreeNode(0, null, root);
        helper(root);
    }

    private void helper(TreeNode root) {
        if (root == null) return;
        TreeNode left = root.left;
        TreeNode right = root.right;
        last.left = null;
        last.right = root;
        last = root;
        helper(left);
        helper(right);
    }
}
```

### 迭代1

迭代写法，这里注意是第三种先序遍历的方法（递归，迭代1，迭代2），注意先入栈右边再入左边，保证左边先处理：

```java
class Solution {
    public void flatten(TreeNode root) {
        helper(root);
    }

    private void helper(TreeNode root) {
        if (root == null) return;
        TreeNode last = new TreeNode(0, null, root);
        Deque<TreeNode> stack = new LinkedList<>();
        stack.push(root);
        while (!stack.isEmpty()) {
            TreeNode top = stack.poll();
            last.left = null;
            last.right = top;
            last = top;
            // 右边先入栈，保证左边先处理
            if (top.right != null) stack.push(top.right);
            if (top.left != null) stack.push(top.left);
        }
    }
}
```

### 迭代2

```java
class Solution {
    public void flatten(TreeNode root) {
        helper(root);
    }

    // 先序遍历树的时候顺序是中左右
    // 如果一个节点的左子节点为空，则为 中、右 ，中的右边就是下一个节点
    // 如果一个节点的左子节点不为空，则为 中、左子节点的最右节点、右
    // 也就是说一个节点的左子节点不为空时，中的右边应该是左子节点的最右节点，然后再跟着右节点
    private void helper(TreeNode root) {
        while (root != null) {
            if (root.left != null) {
                TreeNode next = root.left;
                TreeNode pre = next;
                // 找左子节点的最右节点
                while (pre.right != null) {
                    pre = pre.right;
                }
                // 左子节点的最右节点的下一个为右节点
                pre.right = root.right;
                root.left = null;
                root.right = next;
            }
            root = root.right;
        }
    }
}
```

## 路径总和 III

- 二叉树的节点个数的范围是 `[0,1000]`
- `-10^9 <= Node.val <= 10^9` 
- `-1000 <= targetSum <= 1000` 

给定一个二叉树的根节点 `root` ，和一个整数 `targetSum` ，求该二叉树里节点值之和等于 `targetSum` 的 **路径** 的数目。

**路径** 不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

### 穷举 O(N^2)

访问每一个节点 *node*，检测以 *node* 为起始节点且向下延深的路径有多少种。我们递归遍历每一个节点的所有可能的路径，然后将这些路径数目加起来即为返回结果。

- 定义 helper(p, val) 表示以 p 为起点向下满足和为 val 的路径数目。对每个节点 p 求出 helper(p, targetSum) 就是答案。

  - helper 的实现为

    ```java
        private int helper(TreeNode root, long targetSum) {
            int ret = 0;
            if (root == null) return 0;
            if (root.val == targetSum) ret++;
            ret += helper(root.left, targetSum - root.val);
            ret += helper(root.right, targetSum - root.val);
            return ret;
        }
    ```

- 接下来是对于每个节点都进行 helper，这里先 dfs 遍历所有节点，并对所有节点进行 helper 即可。

```java
class Solution {
    public int pathSum(TreeNode root, long targetSum) {
        return dfs(root, targetSum);
    }

    // 计算所有节点的 helper
    private int dfs(TreeNode root, long targetSum) {
        if (root == null) return 0;
        int ret = 0;
        ret += helper(root, targetSum);
        ret += dfs(root.left, targetSum);
        ret += dfs(root.right, targetSum);
        return ret;
    }

    // 计算以 root 节点向下和为 val 的路径数目
    private int helper(TreeNode root, long targetSum) {
        int ret = 0;
        if (root == null) return 0;
        if (root.val == targetSum) ret++;
        ret += helper(root.left, targetSum - root.val);
        ret += helper(root.right, targetSum - root.val);
        return ret;
    }
}
```

### 前缀和优化 O(N)

我们定义节点的前缀和为：由根结点到当前结点的路径上所有节点的和。

我们利用先序遍历二叉树，记录下根节点 root 到当前节点 p 的路径上除当前节点以外所有节点的前缀和，在已保存的路径前缀和中查找是否存在前缀和刚好等于当前节点到根节点的前缀和 cur 减去 targetSum。

如果 (cur - targetSum) 存在，那么 cur - (cur - targetSum) = targetSum 也存在，就是从某个路径到当前节点的和为 targetSum。

key保存前缀和，value保存对应此前缀和的数量。

然后总体的思路和 560 是一样的。

```java
class Solution {
    public int pathSum(TreeNode root, int targetSum) {
        Map<Long, Integer> prefix = new HashMap<Long, Integer>();
        prefix.put(0L, 1);
        return dfs(root, prefix, 0L, targetSum);
    }

    private int dfs(TreeNode root, Map<Long, Integer> prefix, long cur, int targetSum) {
        if (root == null) return 0;
        int ret = 0;
        // 前缀和
        cur += root.val;

        // 当前从根节点 root 到节点 node 的前缀和为 cur
        // 两节点间的路径和 = 两节点的前缀和之差
        // 查找是否存在 cur - targetSum 的前缀和
        // 如果该前缀和存在，那么 cur - (cur - targetSum) = targetSum 存在
        // 也就是说

        ret = prefix.getOrDefault(cur - targetSum, 0);
        prefix.put(cur, prefix.getOrDefault(cur, 0) + 1);
        ret += dfs(root.left, prefix, cur, targetSum);
        ret += dfs(root.right, prefix, cur, targetSum);
        prefix.put(cur, prefix.getOrDefault(cur, 0) - 1);

        return ret;
    }
}
```

## 二叉树的最近公共祖先

### 暴力递归做法

每个节点判断是不是子节点是不是同时包含 p 和 q

```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        return dfs(root, p, q);
    }

    // 遍历每个节点，当节点的子树同时包含 p 和 q 时，该节点为最近公共祖先
    private TreeNode dfs(TreeNode root, TreeNode p, TreeNode q) {
        if (root == p || root == q) return root;
        if (check(root.left, p, q) && check(root.right, p, q)) return root;
        if (check(root.right, p, q) == false) return dfs(root.left, p, q);
        return dfs(root.right, p, q);
    }

    // root 开始遍历的所有节点中是否有 p 或者 q
    private boolean check(TreeNode root, TreeNode p, TreeNode q) {
        // 如果 p，q 的公共节点为 root，那么从 root 能遍历得到 p 和 q
        if (root == null) return false;
        if (root == p || root == q) return true;
        return check(root.left, p, q) || check(root.right, p, q);
    }
}
```

### 改进递归做法

分为两种情况

- 一种是 p q 在左右不同子树中（都在各自子树中找到的才是最近公共祖先）
- 另一种是在相同子树中（那么先找到的就是最近公共祖先）

```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        return dfs(root, p, q);
    }

    // 对于 root 找 p 和 q
    private TreeNode dfs(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) return null;
        if (root == p || root == q) return root;

        TreeNode left = dfs(root.left, p, q);
        TreeNode right = dfs(root.right, p, q);
        if (left == null) return right;
        if (right == null) return left;
        // 能从 root 找到 p 或者找到 q
        if (left != null && right != null) return root;
        return null;
    }
    
}
```

### 记录父节点的做法

用哈希表记录每个 TreeNode 的父节点，然后遍历 p 的父节点和 q 的父节点，最先出现的那个就是最近公共祖先

```java
class Solution {
    private HashMap<TreeNode, TreeNode> map = new HashMap<>();
    private Set<TreeNode> set = new HashSet<>();

    private void dfs(TreeNode root) {
        if (root == null) return;
        if (root.left != null) {
            map.put(root.left, root);
            dfs(root.left);
        }
        if (root.right != null) {
            map.put(root.right, root);
            dfs(root.right);
        }
    }

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        dfs(root);
        while(p != null) {
            set.add(p); // 要在 get 之前，这样能够处理一个 p 是 q 的父节点的情况
            p = map.get(p);
        }

        while (q != null) {
            if (set.contains(q)) return q;
            q = map.get(q);
        }

        return null;
    }
}
```

## 二叉树中的最大路径和

与二叉树最大直径有异曲同工之妙。

```java
class Solution {
    private int ans = Integer.MIN_VALUE;

    public int maxPathSum(TreeNode root) {
        helper(root);
        return ans;
    }

    private int helper(TreeNode root) {
        if (root == null) return 0;
        int leftMax = Math.max(helper(root.left), 0);
        int rightMax = Math.max(helper(root.right), 0);
        ans = Math.max(leftMax + rightMax + root.val, ans);
        return root.val + Math.max(leftMax, rightMax);
    }
}
```

# 图论

## 岛屿数量

```java
class Solution {
    private int count = 0;
    private boolean[][] visited;
    private int[][] dirs = {
        {0, 1},
        {0, -1},
        {1, 0},
        {-1, 0}
    };

    public int numIslands(char[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        visited = new boolean[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (!visited[i][j] && grid[i][j] == '1') {
                    dfs(grid, i, j);
                    count++;
                }
            }
        }
        return count;
    }

    // 访问所有相邻的岛屿
    // x是行，y是列
    private void dfs(char[][] grid, int x, int y) {    
        visited[x][y] = true;
        for (int i = 0; i < 4; i++) { // 访问所有相邻的
            int dirX = dirs[i][0], dirY = dirs[i][1];
            int m = grid.length;
            int n = grid[0].length;
            int newX = x + dirX, newY = y + dirY;
            if (newX >= 0 && newX < m && newY >= 0 && newY < n && !visited[newX][newY] && grid[newX][newY] == '1') {
                visited[newX][newY] = true; // 原有的连通岛屿都设置成已经访问
                dfs(grid, newX, newY);
            }
        }
    }
}
```

## 腐烂的橘子

### 单源 bfs

不要改动橘子矩阵，新增一个时间矩阵表示橘子的情况。

bfs，如果时间更小则更新时间矩阵。

这个是单源 bfs，需要对于每个腐烂的橘子都进行 bfs。

```java
class Solution {
    private boolean[][] visited;
    private int[][] dirs = {
        {-1, 0},
        {1, 0},
        {0, 1},
        {0, -1},
    }, allTime;
    private int m, n;

    public int orangesRotting(int[][] grid) {
        m = grid.length;
        n = grid[0].length;
        visited = new boolean[m][n];
        allTime = new int[m][n];
        for (int[] row : allTime) {
            Arrays.fill(row, 0x3f3f3f3f);
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                bfs(grid, i, j);
            }
        }
        int maxTime = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] != 0 && allTime[i][j] == 0x3f3f3f3f) { // 存在没腐烂的橘子
                    return -1;
                } else if (grid[i][j] > 0 && allTime[i][j] != 0x3f3f3f3f) {
                    maxTime = Math.max(allTime[i][j], maxTime);
                }
            }
        }
        return maxTime;
    }

    // 返回当前橘子开始最多要多少时间感染所有可感染的橘子
    private void bfs(int[][] grid, int row, int col) {
        if (grid[row][col] != 2) return; // 不是腐烂橘子就不能继续传染
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[]{row, col, 0});
        allTime[row][col] = 0;
        visited[row][col] = true;
        while (!queue.isEmpty()) {
            int[] pos = queue.poll();
            for (int i = 0; i < 4; i++) {
                if (pos[2] == 0x3f3f3f3f) continue; // 不是腐烂橘子就不能继续传染
                int newRow = dirs[i][0] + pos[0];
                int newCol = dirs[i][1] + pos[1];
                int newTime = 1 + pos[2];
                if (ok(newRow, newCol, newTime) && grid[newRow][newCol] > 0) {
                    queue.offer(new int[]{newRow, newCol, newTime});
                    visited[newRow][newCol] = true;
                    allTime[newRow][newCol] = newTime;
                    // grad[newRow][newCol] = 2;
                }
            }
        }
    }

    private boolean ok(int row, int col, int time) {
        return row >= 0 && row < m && col >= 0 && col < n && (!visited[row][col] || time < allTime[row][col]);
    }
}
```

### 多源 bfs

把所有腐烂的橘子都放到队列里面，进行多源 bfs，这样就不用每个腐烂橘子都 bfs 了。

```java
class Solution {
    private boolean[][] visited;
    private int[][] dirs = {
        {-1, 0},
        {1, 0},
        {0, 1},
        {0, -1},
    }, allTime;
    private int m, n;
    Queue<int[]> queue = new LinkedList<>();

    public int orangesRotting(int[][] grid) {
        m = grid.length;
        n = grid[0].length;
        visited = new boolean[m][n];
        allTime = new int[m][n];
        for (int[] row : allTime) {
            Arrays.fill(row, 0x3f3f3f3f);
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 2) {
                    queue.offer(new int[]{i, j, 0});
                    allTime[i][j] = 0;
                    visited[i][j] = true;
                }
            }
        }

        bfs(grid);

        int maxTime = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] != 0 && allTime[i][j] == 0x3f3f3f3f) { // 存在没腐烂的橘子
                    return -1;
                } else if (grid[i][j] > 0 && allTime[i][j] != 0x3f3f3f3f) {
                    maxTime = Math.max(allTime[i][j], maxTime);
                }
            }
        }
        return maxTime;
    }

    // 返回当前橘子开始最多要多少时间感染所有可感染的橘子
    private void bfs(int[][] grid) {
        while (!queue.isEmpty()) {
            int[] pos = queue.poll();
            for (int i = 0; i < 4; i++) {
                int newRow = dirs[i][0] + pos[0];
                int newCol = dirs[i][1] + pos[1];
                int newTime = 1 + pos[2];
                if (ok(newRow, newCol, newTime) && grid[newRow][newCol] > 0) {
                    queue.offer(new int[]{newRow, newCol, newTime});
                    visited[newRow][newCol] = true;
                    allTime[newRow][newCol] = newTime;
                }
            }
        }
    }

    private boolean ok(int row, int col, int time) {
        return row >= 0 && row < m && col >= 0 && col < n && (!visited[row][col] || time < allTime[row][col]);
    }
}
```

## Trie 树

### HashMap 实现

```java
class Trie {
    Map<String, Boolean> map = new HashMap<>();
    Map<String, Boolean> prefixMap = new HashMap<>();

    public Trie() {
        
    }
    
    public void insert(String word) {
        for (int i = 1; i <= word.length(); i++) {
            prefixMap.put(word.substring(0, i), true);
        }
        map.put(word, true);
    }
    
    public boolean search(String word) {
        return map.containsKey(word);
    }
    
    public boolean startsWith(String prefix) {
        return prefixMap.containsKey(prefix);
    }
}
```

### 正常实现

```java
class Trie {
    class TrieNode {
        Map<Character, TrieNode> children = new HashMap<>();
        boolean isEnd = false;
        public TrieNode() {
            
        }
    }

    TrieNode root;
    public Trie() {
        root = new TrieNode();
    }
    
    public void insert(String word) {
        TrieNode node = root;
        for (char ch : word.toCharArray()) {
            node = node.children.computeIfAbsent(ch, v -> new TrieNode());
        }
        node.isEnd = true;
    }
    
    public boolean search(String word) {
        TrieNode node = root;
        for (char ch : word.toCharArray()) {
            node = node.children.get(ch);
            if (node == null) return false;
        }
        return node.isEnd;
    }
    
    public boolean startsWith(String prefix) {
        TrieNode node = root;
        for (char ch : prefix.toCharArray()) {
            node = node.children.get(ch);
            if (node == null) return false;
        }
        return true;
    }
}
```



# 回溯

## 全排列

主要就是回溯前后的状态的更新和恢复

```java
class Solution {
    private int n;
    private boolean[] visited;
    private List<List<Integer>> ans;
    private List<Integer> tmp;

    public List<List<Integer>> permute(int[] nums) {
        n = nums.length;
        visited = new boolean[n];
        ans = new ArrayList<>();
        tmp = new ArrayList<>();
        dfs(nums, 0);
        return ans;
    }

    private void dfs(int[] nums, int times) {
        if (times == n) {
            ans.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                tmp.add(nums[i]);
                visited[i] = true;
                dfs(nums, times + 1);
                visited[i] = false;
                tmp.remove(tmp.size() - 1);
            }
        }
    }
}
```





# 子串

## 和为 K 的子数组

2 * 10^4 O(n^2) 或者 O(n)

### 前缀和

```
class Solution {
    public int subarraySum(int[] nums, int k) {
        int left = 0, n = nums.length, right = n - 1, ans = 0;;
        int[] sum = new int[n];
        sum[0] = nums[0];
        for (int i = 1; i < n; i++) {
            sum[i] = sum[i - 1] + nums[i];
        }
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (sum[j] - sum[i] == k) { // 这种写法无法计算前 n 个的和
                    ans++;
                }
            }
            
        }
        for (int i = 0; i < n; i++) { // 补上
            if (sum[i] == k) {
                ans++;
            }
        }
        return ans;
    }
}
```



## 优化前缀和中的 O(n^2)

用 "两数之和" 的思路来优化掉枚举所有前缀和的过程，用 HashMap 直接找到想要的前缀和。

```java
class Solution {
    public int subarraySum(int[] nums, int k) {
        int left = 0, n = nums.length, right = n - 1, ans = 0;;
        int[] sum = new int[n];
        sum[0] = nums[0];
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 1; i < n; i++) {
            sum[i] = sum[i - 1] + nums[i];
        }
        // 方法一：暴力前缀和 

        // for (int i = 0; i < n; i++) {
        //     for (int j = i + 1; j < n; j++) {
        //         if (sum[j] - sum[i] == k) { // 这种写法无法计算前 n 个的和
        //             ans++;
        //         }
        //     }
        // }

        // 方法一的实质是求两数之差为固定值的数有多少
        // sum[j] - sum[i] = k
        // 用 "两数之和" 的方法用哈希表进行优化 a + b = target -> a = target - b
        for (int i = 0; i < n; i++) {
            int num = k + sum[i];
            if (map.containsKey(sum[i])) { // 如果需要的正好有
                ans += map.get(sum[i]);
            }
            map.put(num, map.getOrDefault(num, 0) + 1); // 需要 k + sum[i]
        }

        for (int i = 0; i < n; i++) { // 补上
            if (sum[i] == k) {
                ans++;
            }
        }
        return ans;
    }
}
```

## 滑动窗口最大值

10^5 O(n) 可以做

给你一个整数数组 `nums`，有一个大小为 `k` 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 `k` 个数字。滑动窗口每次只向右移动一位。

返回 *滑动窗口中的最大值* 。

像前缀和一样，维护区间的最大值呢？

### ST 表

复杂度为 O(nlogn)：

```java
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        int n = nums.length, l = 0, logN = (int)(Math.log(n) / Math.log(2)) + 1;
        int[] result = new int[n - k + 1];
        int f[][] = new int[n + 1][logN];
        int[] logn = new int[n + 5];

        for (int i = 1; i <= n; i++) {
            f[i][0] = nums[i - 1];
        }

        // pre
        logn[1] = 0;
        logn[2] = 1;
        for (int i = 3; i < n; i++) {
            logn[i] = logn[i / 2] + 1;
        }

        for (int j = 1; j <= logN; j++)
            for (int i = 1; i + (1 << j) - 1 <= n; i++)
                f[i][j] = Math.max(f[i][j - 1], f[i + (1 << (j - 1))][j - 1]);  // ST表具体实现
        for (int i = 1; i <= n - k + 1; i++) {
            int x = i, y = Math.min(n, i + k - 1);
            int s = logn[y - x + 1];
            result[i-1] = Math.max(f[x][s], f[y - (1 << s) + 1][s]);
        }
        return result;
    }
}
```

### 优先队列

O(nlogn)

这里的思路是用一个堆来维护最大值，同时在堆中记录下最大值的下标，当左指针移动时，要删掉那些失效的最大值。

```java
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        PriorityQueue<Map.Entry<Integer, Integer>> pq = new PriorityQueue<>((a, b) -> b.getKey() - a.getKey());
        int n = nums.length, l = 0;
        int ret[] = new int[n - k + 1];
        for (int r = 0; r < n; r++) {
            pq.offer(new java.util.AbstractMap.SimpleEntry<>(nums[r], r));
            if (r - l + 1 == k) {
                while (pq.peek().getValue() < l) pq.poll();
                ret[r - k + 1] = pq.peek().getKey();
                l++;
            }
        }
        return ret;
    }
}
```

### 单调队列

```java
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        int n = nums.length, l = 0;
        int[] ans = new int[n - k + 1];
        Deque<Integer> deque = new LinkedList<Integer>(); // 递增存储下标
        for (int r = 0; r < n; r++) {
            while (!deque.isEmpty() && nums[r] > nums[deque.peekLast()]) { // 放进去的是保持下标递增的情况下值最大的
                deque.pollLast();
            }
            deque.offerLast(r);
            if (r - l + 1 == k) {
                ans[l] = nums[deque.peekFirst()];
                while (!deque.isEmpty() && deque.peekFirst() <= l) { // 淘汰掉滑动窗口以外的
                    deque.pollFirst();
                }
                l++;
            }
        }
        return ans;
    }
}
```

### 分块 + 前后缀数组

```java
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        int n = nums.length;
        if (n == 1) return nums;
        int[] prefixMax = new int[n]; // 以 i 结尾的前缀最大值，也就是查询的左边
        int[] suffixMax = new int[n]; // 以 i 开头的后缀最大值，也就是查询的右边
        // [ a, b, c ] [ d, e, f ]
        // 按 k 分块，如果是边界，那么需要第一个的后缀最大值和第二个的前缀最大值拼起来
        // 如果不是边界，那么直接取后缀最大值就好
        for (int i = 0; i < n; i++) {
            if (i % k == 0) { // 边界
                prefixMax[i] = nums[i];

            } else { // 在块中间
                prefixMax[i] = Math.max(prefixMax[i - 1], nums[i]);
            }
        }
        for (int i = n - 1; i >= 0; i--) {
            if ((i + 1) % k == 0 || i == n - 1) {
                suffixMax[i] = nums[i];
            } else {
                suffixMax[i] = Math.max(suffixMax[i + 1], nums[i]);
            }
        }

        int[] ans = new int[n - k + 1];
        // [ a, b, c ] [ d, e, f ]
        for (int i = 0 ; i < n - k + 1; i++) { // 滑动窗口开始位置
            if (i % k == 0) { // 边界
                ans[i] = prefixMax[i + k - 1];
            } else { // [ a, b, c ] [ d, e, f ]
                     // b 为滑动窗口开始的时候，元素为 b, c, d，最大值为 max([b, c], [d])
                     // -> max(suffixMax[1], prefixMax[3])
                ans[i] = Math.max(suffixMax[i], prefixMax[i + k - 1]);
            }
        }
        return ans;


    }
}
```



## 双指针
**前后指针：**经典的一个 pre 指针，一个 cur 指针：可以解决反转链表、交换节点等问题。
**快慢指针：**还有一个 fast 指针，一个 slow 指针：可以解决删除第 n 个元素的问题。

19.删除链表的倒数第 N 个结点

142.环形链表

# 哈希表

## 两数之和

10^4 O(n)

一个哈希表

## 三数之和

3000

O(n^2)

排序+双指针

## 四数之和

排序+双指针

注意溢出

## 四数相加

与上一题不同在于有 4 个数组，4 个数组等长度，上一题每个区间长度不同

哈希表 + 哈希表

## 字母异位词分组

10^4 O(n) 或 O(nlogn)

主要考虑异位词表示为相同的 map key，这样就可以将异位词聚集在一起。

## 最长连续序列

10^5 O(n)

未排序的数组，O(n) 找到数字连续的最长序列，不要求在原数组中连续。

排序做法为 O(nlogn)

key：考虑某一个数是不是连续序列的第一个数字，如果是则继续往下。

```java
        for (int num : set) {
            int i = 1;
            while (set.contains(num + i)) {
                i++;
                longest = Math.max(longest, i);
            }
        }
```

这种写法最坏会变成 O(n^2)，需要思考如何跳过重复情况。如果再开一个 TreeSet 来定位下一个数字是 O(logn)，应该可以。

```java
class Solution {
    public int longestConsecutive(int[] nums) {
        if (nums.length == 0) return 0;
        Set<Integer> set = new HashSet<>();
        TreeSet<Integer> treeSet = new TreeSet<>();
        Integer next = 0x3f3f3f3f;
        for (int i = 0; i < nums.length; i++) {
            set.add(nums[i]);
            treeSet.add(nums[i]);
            next = Math.min(next, nums[i]);
        }
        int longest = 1;
        while (next != null) {
            int i = 1;
            while (set.contains(next)) {
                next = next + 1;
                longest = Math.max(longest, i);
                i++;
            }
            next = treeSet.higher(next); // 定位下一个数字
        }

        return longest;
    }
}
```

还有一种 O(1) 定位下一个数字的方法：

如果这个数字为 x，那么不存在 x-1 的话，这个数字一定是连续序列的第一个数字。

```java
class Solution {
    public int longestConsecutive(int[] nums) {
        if (nums.length == 0) return 0;
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < nums.length; i++) {
            set.add(nums[i]);
        }
        int longest = 1;
        for (int num : set) {
            if (set.contains(num - 1)) continue;
            int i = 1;
            while (set.contains(num + i)) {
                i++;
                longest = Math.max(longest, i);
            }
        }

        return longest;
    }
}
```



# 字符串

## KMP 算法

next 数组：**是一个前缀表，前缀表是用来回退的，它记录了模式串与主串(文本串)不匹配的时候，模式串应该从哪里开始重新匹配。**

那么什么是前缀表：**记录下标i之前（包括i）的字符串中，有多大长度的相同前缀后缀。**

**下标5之前这部分的字符串（也就是字符串aabaa）的最长相等的前缀 和 后缀字符串是 子字符串aa ，因为找到了最长相等的前缀和后缀，匹配失败的位置是后缀子串的后面，那么我们找到与其相同的前缀的后面重新匹配就可以了。**



## 反转字符串中的单词

方法一：split 之后拼接。

```java
"a good   example".split() // [a, good, , , example]，分割后存在 ""
```

方法二：反转整个字符串之后，再反转单个字符串。

## 最长公共子串

状态转移方程如下，dp\[i\]\[j\] 表示字符串 x 以 i 结尾，字符串 y 以 j 结尾的最长公共子串，这样就有了：
$$
d p[i][j]=\left\{\begin{array}{l}
d p[i-1][j-1]+1, \text { 当且仅当 } x[i]=y[j] \\
0, \text { 当 } x[i] \ne y[j]
\end{array}\right.
$$
按照上面方程实现的算法时间复杂度为 $O(n^2)$，空间复杂度为 $O(n^2)$。

![image.png](../leetcode/d6f0b0e17ed6e13f5c042d172b1ddca782cb6aba589f5fcfea8944831614502f-image.png)

注意到，更新 $dp[i][j]$ 只需要上一列，即 $dp[i-1]$ 列，所以可以将空间复杂度降低为 $O(n)$，但是需要注意因为使用的是相同的数组列，所以字符串不相等时需要设置 $dp[j] = 0$，同时要注意从后向前更新数组，因为如果从前向后更新，那么当前的 $dp[j]$ 使用的是当前列刚刚更新过的数据，而我们需要的是上一列的数据，所以可以从后向前更新数据避免这个问题。

rust 代码如下：

```rust
let mut dp = vec![0; s2.len()];
for i in 0..s1.len() {
    // 逆序迭代是因为更新a[i][j]需要a[i-1][j-1]
    // 现在是一个数组，所以 a[j] 是原来的 a[i][j]，而我们需要的是 a[i-1][j]
    // 所以从后向前迭代，a[j] 是原来的 a[i-1][j]
    for j in (0..s2.len()).s2() {
        if s[i] == s2[j] {
            if i == 0 || j == 0 {
                dp[j] = 1;
            } else {
                dp[j] = dp[j - 1] + 1;
            }
            if dp[j] > max_len {
                let before_s2 = s2.len() - 1 - j;
                if before_s2 + dp[j] - 1 == i {
                    max_len = dp[j];
                    max_end = i;
                }
            }
        } else {
            // 与之前不同，之前使用的是不同的列，所以不需要置0
            dp[j] = 0;
        }
    }
}
```

## 最长回文子串

### 动态规划

将字符串倒置之后求最长公共子串（状态转移方程与最长公共子串相同），并判断是否为回文子串，这里回文子串「由倒置字符串推出的原字符串末尾下标」与「i」应该相等。

代码中 `longest_palindrome1` 的求最长公共子串空间复杂度为 $O(n^2)$，`longest_palindrome2` 的求最长公共子串空间复杂度为 $O(n)$。

代码如下：

```rust
struct Solution;

impl Solution {
    pub fn longest_palindrome1(s: String) -> String {
        if s.len() <= 1 {
            return s;
        }
        let rev: String = s.chars().rev().collect();
        let rev = rev.as_bytes();
        let s = s.as_bytes();
        let mut dp = vec![vec![0; rev.len()]; s.len()];
        let mut max_len = 0;
        let mut max_end = 0;
        for i in 0..s.len() {
            for j in 0..rev.len() {
                if s[i] == rev[j] {
                    if i == 0 || j == 0 {
                        dp[i][j] = 1;
                    } else {
                        dp[i][j] = dp[i - 1][j - 1] + 1;
                    }
                }
                if dp[i][j] > max_len {
                    // 如果是回文串，那么「由倒置字符串推出的原字符串末尾下标」与「i」应该相等
                    // 其中，倒置字符串的 rev.len() - 1 - j，也就是倒置之前的开始下标，减一是因为长度比下标多一
                    // 再加上 dp[i][j] - 1，就是原字符串的末尾下标。abc，a的下标为0，长度为3，0+3为3，但是最大下标为2，所以需要减一
                    let before_rev = rev.len() - 1 - j;
                    if before_rev + dp[i][j] - 1 == i {
                        max_len = dp[i][j];
                        max_end = i;
                    }
                }
            }
        }
        std::str::from_utf8(&s[max_end + 1 - max_len..max_end + 1])
            .unwrap()
            .to_string()
    }
    pub fn longest_palindrome2(s: String) -> String {
        if s.len() < 1 {
            return s;
        }
        let rev: String = s.chars().rev().collect();
        let s = s.as_bytes();
        let rev = rev.as_bytes();
        let mut max_len = 0;
        let mut max_end = 0;
        let mut dp = vec![0; rev.len()];
        for i in 0..s.len() {
            // 逆序迭代是因为更新a[i][j]需要a[i-1][j-1]
            // 现在是一个数组，所以 a[j] 是原来的 a[i][j]，而我们需要的是 a[i-1][j]
            // 所以从后向前迭代，a[j] 是原来的 a[i-1][j]
            for j in (0..rev.len()).rev() {
                if s[i] == rev[j] {
                    if i == 0 || j == 0 {
                        dp[j] = 1;
                    } else {
                        dp[j] = dp[j - 1] + 1;
                    }
                    if dp[j] > max_len {
                        let before_rev = rev.len() - 1 - j;
                        if before_rev + dp[j] - 1 == i {
                            max_len = dp[j];
                            max_end = i;
                        }
                    }
                } else {
                    // 与之前不同，之前使用的是不同的列，所以不需要置0
                    dp[j] = 0;
                }
            }
        }
        std::str::from_utf8(&s[max_end + 1 - max_len..max_end + 1])
            .unwrap()
            .to_string()
    }
}
```

### 中心拓展算法

为了避免在之后的叙述中出现歧义，这里我们指出什么是“朴素算法”。

该算法通过下述方式工作：对每个中心位置 $i$ 在比较一对对应字符后，只要可能，该算法便尝试将答案加 $1$。

该算法是比较慢的：它只能在 $O(n^2)$ 的时间内计算答案。

该算法的实现如下：

```c++
// C++ Version
vector<int> d1(n), d2(n);
for (int i = 0; i < n; i++) {
  d1[i] = 1;
  while (0 <= i - d1[i] && i + d1[i] < n && s[i - d1[i]] == s[i + d1[i]]) {
    d1[i]++;
  }

  d2[i] = 0;
  while (0 <= i - d2[i] - 1 && i + d2[i] < n &&
         s[i - d2[i] - 1] == s[i + d2[i]]) {
    d2[i]++;
  }
}
```

```python
# Python Version
d1 = [0] * n
d2 = [0] * n
for i in range(0, n):
    d1[i] = 1
    while 0 <= i - d1[i] and i + d1[i] < n and s[i - d1[i]] == s[i + d1[i]]:
        d1[i] += 1

    d2[i] = 0
    while 0 <= i - d2[i] - 1 and i + d2[i] < n and s[i - d2[i] - 1] == s[i + d2[i]]:
        d2[i] += 1
```

### Manacher 算法[^1][^2]

Manacher 算法是对中心拓展算法的优化，为了快速计算，我们维护已找到的最靠右的子回文串的 **边界 $(l, r)$**（即具有最大 $r$ 值的回文串，其中 $l$ 和 $r$ 分别为该回文串左右边界的位置）。初始时，我们置 $l = 0$ 和 $r = -1$（*-1*需区别于倒序索引位置，这里可为任意负数，仅为了循环初始时方便）。

现在假设我们要对下一个 $i$ 计算 $P[i]$，而之前所有 $P[]$ 中的值已计算完毕。我们将通过下列方式计算：

- 如果 $i$ 位于当前子回文串之外，即 $i > r$，那么我们调用朴素算法。

  因此我们将连续地增加 $d_1[i]$，同时在每一步中检查当前的子串 $[i - P[i] \dots i +  P[i]]$（$P[i]$ 表示半径长度，下同）是否为一个回文串。如果我们找到了第一处对应字符不同，又或者碰到了 $s$  的边界，则算法停止。在两种情况下我们均已计算完 $P[i]$。此后，仍需记得更新 $(l, r)$。

- 现在考虑 $i \le r$ 的情况。我们将尝试从已计算过的 $P[]$ 的值中获取一些信息。首先在子回文串  $(l, r)$ 中反转位置 $i$，即我们得到 $j = l + (r - i)$。现在来考察值 $P[j]$。因为位置 $j$ 同位置  $i$ 对称，我们 **几乎总是** 可以置 $P[i] = P[j]$。

  存在 **棘手的情况**，主要有以下：

  - 超出了 $r$
  
    ![图转自 LeetCode](../leetcode/b0d52a5f30747e55ef09b3c7b7cfc23026e37040edc41f387263e8f8a0ba8f49-image.png)
  
    当我们要求 $P [ i ]$ 的时候，$P [mirror] = 7$，而此时 $P [ i ]$ 并不等于 $7$，为什么呢，因为我们从 $i$ 开始往后数 $7$ 个，等于 $22$，已经超过了最右的 $r$，此时不能利用对称性了，但我们一定可以扩展到 $r$ 的，所以 $P [ i ]$ 至少等于 $r - i = 20 - 15 = 5$，会不会更大呢，我们只需要比较 $T [ r+1 ]$ 和 $T [ r+1 ]$ 关于 $i$ 的对称点就行了，就像中心扩展法一样一个个扩展。
  
  - $P[i]$ 遇到了原字符串的左边界
  
    ![image.png](../leetcode/714e6f768e67304fb7162ecac3ae85fcf23ad82a21456e8ca55ac2c8cfd2609e-image.png)
  
    此时$P [ i_{mirror} ] = 1$，但是 $P [ i ]$ 赋值成 1 是不正确的，出现这种情况的原因是 $P [ i_{mirror} ]$ 在扩展的时候首先是 "#" == "#"，之后遇到了 "^" 和另一个字符比较，也就是到了边界，才终止循环的。而 $P [ i ]$ 并没有遇到边界，所以我们可以继续通过中心扩展法一步一步向两边扩展就行了。
  
  - $i = r$
  
    此时我们先把 P [ i ] 赋值为 0，然后通过中心扩展法一步一步扩展就行了。
  
  考虑 $r$ 的更新
  
  就这样一步一步的求出每个 $P [ i ]$，当求出的 $P [ i ]$ 的右边界大于当前的 $r$ 时，我们就需要更新 $r$ 为当前的回文串了。

## 最长公共子序列（LCS）

### 动态规划

状态转移方程如下：
$$
d p[i][j]=\left\{\begin{array}{ll}
d p[i-1][j-1]+1, & t e x t_{1}[i-1]=t e x t_{2}[j-1] \\
\max (d p[i-1][j], d p[i][j-1]), & t e x t_{1}[i-1] \neq t e x t_{2}[j-1]
\end{array}\right.
$$
LCS 对应的状态转移方程与最长公共子串不同之处在于：

- 最长公共子串要求字符串连续，所以下一个状态只能由上一个对应的字符串得到。
- LCS 不要求字符串连续，所以可以前后移动，就有了第二个式子。

知道状态定义之后，我们开始写状态转移方程。

- 当 $text_1[i - 1] = text_2[j - 1]$ 时，说明两个子字符串的最后一位相等，所以最长公共子序列又增加了 1，所以 $dp[i][j] = dp[i - 1][j - 1] + 1$；举个例子，比如对于 `ac` 和 `bc` 而言，他们的最长公共子序列的长度等于 `a` 和 `b` 的最长公共子序列长度 $0 + 1 = 1$。

- 当 $text_1[i - 1] \ne text_2[j - 1]$ 时，说明两个子字符串的最后一位不相等，那么此时的状态 $dp[i][j]$ 应该是 $dp[i - 1][j]$ 和 $dp[i][j - 1]$ 的最大值。举个例子，比如对于 `ace` 和 `bc` 而言，他们的最长公共子序列的长度等于

   ① `ace` 和 `b` 的最长公共子序列长度 `0 ` 与

  ② `ac` 和 `bc` 的最长公共子序列长度 `1` 的最大值，即 `1`。

代码如下：

```rust
struct Solution;

impl Solution {
    pub fn longest_common_subsequence(text1: String, text2: String) -> i32 {
        let text1 = text1.as_bytes();
        let text2 = text2.as_bytes();
        let m = text1.len();
        let n = text2.len();
        // dp[i][j] 代表 text1[0..i] 与 text2[0..j] 的最大子序列，注意不包括第 i 和第 j 个字符
        // 同理，dp 数组要循环到 m 与 n 才结束
        let mut dp = vec![vec![0; n + 1]; m + 1];
        for i in 1..=m {
            for j in 1..=n {
                // 这里要注意，比较的是第 i-1 与第 j-1 个字符
                if text1[i - 1] == text2[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = std::cmp::max(dp[i][j - 1], dp[i - 1][j]);
                }
            }
        }
        return dp[m][n];
    }
}
```

# 动态规划

对于动态规划，可以先初始化一个 dp 数组，然后手写出 dp[0] dp[1] dp[2] dp[3] 等等

## 杨辉三角

主要难点在处理边界情况

```java
class Solution {
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> ret = new ArrayList<>();
        for (int i = 1; i <= numRows; i++) {
            List<Integer> arr = new ArrayList<>(i);
            for (int j = 0; j < i; j++) {
                arr.add(0);
            }
            arr.set(0, 1);
            arr.set(i - 1, 1);
            List<Integer> lastArr;
            if (ret.size() >= 2) {
                lastArr = ret.get(ret.size() - 1);
                for (int j = 1; j < i - 1; j++) { // 跳过第一个和最后一个
                    arr.set(j, lastArr.get(j-1) + lastArr.get(j));
                }
            }
            ret.add(arr);
        }
        return ret;
    }
}
```

## 打家劫舍

要点在于对于某一个房子是否抢劫以及边界处理。

```java
class Solution {
    public int rob(int[] nums) {
        int n = nums.length;
        if (n == 1) return nums[0];
        if (n == 2) return Math.max(nums[0], nums[1]);
        int[] dp = new int[n + 1];
        dp[0] = 0;
        dp[1] = nums[0];
        dp[2] = Math.max(nums[0], nums[1]);
        for (int i = 3; i <= n; i++) {
            dp[i] = Math.max(dp[i - 2] + nums[i - 1], dp[i - 1]);
        }
        return dp[n];
    }
}
```

## 最大子序和

要点在于当前数字结尾的最大子串的和只能来自于前一个，也就是上楼梯只能由前一个上过来，或者现在新开一个。

```java
class Solution {
    public int maxSubArray(int[] nums) {
        int n = nums.length;
        int[] dp = new int[n + 1]; // 第 i 个数结尾的子数组最大值
        int ans = -0x3f3f3f3f;
        for (int i = 1; i <= n; i++) {
            dp[i] = Math.max(dp[i - 1] + nums[i - 1], nums[i - 1]);
            ans = Math.max(ans, dp[i]);
        }
        return ans;
    }
}
```



## 完全平方数

### 动态规划

要点是 dp[i] 表示结果为 i 的最少数量，然后转移的话，dp[i] 只能从 dp[i - 所有平方数] 来，于是写出状态转移方程。

```java
class Solution {
    public int numSquares(int n) {
        if (n == 1) return 1;
        if (n == 2) return 2;
        if (n == 3) return 3;
        if (n == 4) return 1;
        if (n == 5) return 2;
        int sqrtN = (int)Math.sqrt(n) + 1;
        int[] square = new int[sqrtN];
        for (int i = 1; i < sqrtN; i++) {
            square[i] = i * i;
        }
        int[] dp = new int[n + 1];
        dp[0] = 0;
        dp[1] = 1;
        dp[2] = 2;
        dp[3] = 3;
        dp[4] = 1;
        dp[5] = 2;

        for (int i = 1; i <= n; i++) {
            int min = 999;
            for (int j = 1; j < sqrtN; j++) {
                if (i - square[j] >= 0) {
                    min = Math.min(dp[i - square[j]] + 1, min);
                }
            }
            dp[i] = min;
        }

        return dp[n];
    }
}
```

优化一下

```java
class Solution {
    public int numSquares(int n) {
        int[] dp = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            int min = 999;
            for (int j = 1; j * j <= i; j++) {
                if (i - j * j >= 0) {
                    min = Math.min(dp[i - j * j] + 1, min);
                }
            }
            dp[i] = min;
        }

        return dp[n];
    }
}
```

### 四平方和定理

![image-20250107155619501](/image-20250107155619501.png)





## 零钱兑换

这是一个完全背包问题，硬币可以重复使用。与爬楼梯类似。

$dp[i] = dp[i - coins[j]] + 1$。

```
class Solution {
    public int coinChange(int[] coins, int amount) {
        if (amount == 0) return 0;
        int n = coins.length;
        int[] dp = new int[Math.max(n, amount) + 10]; // 可以凑成 n 所需的最少的硬币个数
        // 与爬楼梯一样，dp[i] 可以从 dp[i - coins[j]] 过来
        for (int i = 0; i < n; i++) {
            if (coins[i] > amount) {
                continue;
            }
            dp[coins[i]] = 1;
        }
        for (int i = 0; i <= amount; i++) {
            if (dp[i] == 0) {
                dp[i] = 0x3f3f3f3f;
            }
        }
        dp[0] = 0;
        int max = 0;
        for (int i = 1; i <= amount; i++) {
            for (int j = 0; j < n; j++) {
                if (i - coins[j] >= 0) {
                    dp[i] = Math.min(dp[i - coins[j]] + 1, dp[i]);
                }
            }
        }

        return dp[amount] == 0x3f3f3f3f ? -1 : dp[amount];
    }
}
```



## 单词拆分

- `1 <= s.length <= 300`
- `1 <= wordDict.length <= 1000`
- `1 <= wordDict[i].length <= 20`

### 模拟 + 剪枝

不断地尝试所有可能的拼接，看最后能否拼接出来，注意要剪枝，否则会超时。

```java
class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        int n = wordDict.size();
        Set<String> dp = new HashSet<>();
        int maxLen = 0x3f3f3f3f;
        for (int i = 0; i < n; i++) {
            String word = wordDict.get(i);
            if (s.startsWith(word)) // 剪枝
                dp.add(word);
            maxLen = Math.min(maxLen, word.length());
        }


        while (maxLen < s.length()) {
            Set<String> tmp = new HashSet<>(dp); // 创建一个临时副本用于遍历
            int minLen = 0x3f3f3f3f;
            for (String cur : tmp) {
                if (cur.length() < maxLen) continue;
                for (int i = 0; i < n; i++) {
                    String newString = cur + wordDict.get(i);
                    if (s.startsWith(newString)) { // 剪枝
                        minLen = Math.min(newString.length(), minLen);
                        dp.add(newString);
                    }
                }
            }
            maxLen = Math.max(maxLen, minLen);
        }
        return dp.contains(s);
    }
}
```

优化一下

### 动态规划

![image-20250107182951481](/image-20250107182951481.png)

```java
class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        int n = wordDict.size();
        Set<String> set = new HashSet<>(wordDict);
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && set.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[s.length()];
    }
}
```

## 最长递增子序列

要点在于 dp 为以第 i 个数字结尾的最长上升子序列长度，也就是 dp[i] 是 i 被选择的情况下的最长上升子序列长度。

这题要求的是子序列，而不是子数组，也就是可以跳过一些数字，所以动态转移方程就是对于 dp[i] 可以从任意的 dp[i - j] 调过来。

```java
class Solution {
    public int lengthOfLIS(int[] nums) {
        int n =  nums.length, max = 1;
        int[] dp = new int[n + 1]; // 以第 i 个数字结尾的最长上升子序列长度
        for (int i = 1; i <= n; i++) {
            dp[i] = 1; // 如果选当前的 i，至少有 1 个长度
            for (int j = 1; j < i; j++) {
                if (nums[i - 1] > nums[j - 1]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            max = Math.max(max, dp[i]);
        }
        return max;
    }
}
```

## 乘积最大子数组

`dp_max[i]` 表示以第 `i` 个元素结尾的连续子数组的最大乘积。

对于每一个 dp[i] 来说，他会从前一个最大的 dp[i - 1] 过来，或者从最小的 dp[i - 1] 过来，或者在当前位置另起炉灶。与子序列不同，dp[i] 的子数组只能从 dp[i - 1] 过来，而不能从任意的 dp[i - j] 过来。

```java
class Solution {
    public int maxProduct(int[] nums) {
        if (nums.length == 0) return 0;
        int n = nums.length;
        int[] dp_max = new int[n];
        int[] dp_min = new int[n];
        dp_max[0] = nums[0];
        dp_min[0] = nums[0];
        int ans = nums[0];
        for (int i = 1; i < n; i++) {
            dp_max[i] = Math.max(Math.max(dp_max[i-1] * nums[i], dp_min[i-1] * nums[i]), nums[i]);
            dp_min[i] = Math.min(Math.min(dp_max[i-1] * nums[i], dp_min[i-1] * nums[i]), nums[i]);
            ans = Math.max(ans, dp_max[i]);
        }
        return ans;
    }
}
```

## 分割等和子集

`1 <= nums.length <= 200`，`1 <= nums[i] <= 100` 数据范围暗示了是和数组中最大数有关的二维 dp

给你一个 **只包含正整数** 的 **非空** 数组 `nums` 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

如何将该问题转换为具有最优子结构的问题是难点。

动态规划，时间复杂度与元素大小相关的。

这个问题可以转换为：给定一个只包含正整数的非空数组 nums[0]，判断是否可以从数组中选出一些数字，使得这些数字的和等于整个数组的元素和的一半。因此这个问题可以转换成「0−1 背包问题」。这道题与传统的「0−1 背包问题」的区别在于，传统的「0−1 背包问题」要求选取的物品的重量之和不能超过背包的总容量，这道题则要求选取的数字的和恰好等于整个数组的元素和的一半。类似于传统的「0−1 背包问题」，可以使用动态规划求解。

关键在于：**将这个数组分割成两个子集 = 选出的数字的和是数组一半**

然后关键的转移方程为：

1. 和为 j 的数字可以由 j - nums[i]（如果存在的话） 得到
2. 和为 j 的数字可以由 j （如果存在的话）得到

```java
class Solution {
    public boolean canPartition(int[] nums) {
        int n = nums.length, target = 0, max = 0;
        for (int i = 0; i < n; i++) {
            target += nums[i];
            max = Math.max(max, nums[i]);
        }
        if ((target & 1) == 1) return false; // 奇数
        else target /= 2;
        if (max > target) return false;

        boolean[][] dp = new boolean[n][target + 1]; // 0..i 的数字中是否存在方案使得和为 j
        // 数字 nums[i] 所在的行都能使得和为 nums[i]
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                dp[i][nums[j]] = true;
            }
        }

        // 0..i 中怎么选都能使得和为 0
        for (int i = 0; i < n; i++) {
            dp[i][0] = true;
        }

        // 和为 j 的数字可以由 j - nums[i]（如果存在的话） 得到
        // 和为 j 的数字可以由 j （如果存在的话）得到
        // 可以根据上面的内容对 dp 数组进行填表
        for (int i = 1; i < n; i++) {
            for (int j = 1; j <= target; j++) {

                if (j >= nums[i]) {

                    dp[i][j] = dp[i - 1][j] | dp[i - 1][j - nums[i]];

                    // 与上面的内容等价
                    // if (dp[i - 1][j - nums[i]]) {
                    //     dp[i][j] = true;
                    // }
                    // if (dp[i - 1][j]) {
                    //     dp[i][j] = true;
                    // }

                } else {
                    dp[i][j] = dp[i - 1][j];

                    // 与上面的内容等价
                    if (dp[i - 1][j]) {
                        dp[i][j] = true;
                    }
                }
            }
        }

        
        return dp[n - 1][target];
    }
}
```

然后又有这一行仅仅由上一行确定得到，于是有：

```java
class Solution {
    public boolean canPartition(int[] nums) {
        int n = nums.length, target = 0, max = 0;
        for (int i = 0; i < n; i++) {
            target += nums[i];
            max = Math.max(max, nums[i]);
        }
        if ((target & 1) == 1) return false; // 奇数
        else target /= 2;
        if (max > target) return false;

        boolean[][] dp = new boolean[n][target + 1]; // 0..i 的数字中是否存在方案使得和为 j
        // 数字 nums[i] 所在的行都能使得和为 nums[i]
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                dp[i][nums[j]] = true;
            }
        }

        // 0..i 中怎么选都能使得和为 0
        for (int i = 0; i < n; i++) {
            dp[i][0] = true;
        }

        // 和为 j 的数字可以由 j - nums[i]（如果存在的话） 得到
        // 和为 j 的数字可以由 j （如果存在的话）得到
        // 可以根据上面的内容对 dp 数组进行填表
        // for (int i = 1; i < n; i++) {
        //     for (int j = 1; j <= target; j++) {

        //         if (j >= nums[i]) {

        //             dp[i][j] = dp[i - 1][j] | dp[i - 1][j - nums[i]];

        //             // 与上面的内容等价
        //             // if (dp[i - 1][j - nums[i]]) {
        //             //     dp[i][j] = true;
        //             // }
        //             // if (dp[i - 1][j]) {
        //             //     dp[i][j] = true;
        //             // }

        //         } else {
        //             dp[i][j] = dp[i - 1][j];

        //             // 与上面的内容等价
        //             if (dp[i - 1][j]) {
        //                 dp[i][j] = true;
        //             }
        //         }
        //     }
        // }

        for (int i = 1; i < n; i++) {
            int num = nums[i];
            for (int j = 1; j <= target; j++) {
                if (j >= num) {
                    dp[i][j] = dp[i - 1][j] | dp[i - 1][j - num];
                } else {
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }

        
        return dp[n - 1][target];
    }
}
```

优化空间：

```java
class Solution {
    public boolean canPartition(int[] nums) {
        int n = nums.length, target = 0, max = 0;
        for (int i = 0; i < n; i++) {
            target += nums[i];
            max = Math.max(max, nums[i]);
        }
        if ((target & 1) == 1) return false; // 奇数
        else target /= 2;
        if (max > target) return false;

        boolean[] dp = new boolean[target + 1]; // 0..i 的数字中是否存在方案使得和为 j

        dp[0] = true;

        for (int i = 0; i < n; i++) {
            int num = nums[i];
            for (int j = target; j >= num; j--) {
                dp[j] |= dp[j - num];
            }
        }

        
        return dp[target];
    }
}
```

## 最长有效括号

### 动态规划

`0 <= s.length <= 3 * 10^4`

给你一个只包含 `'('` 和 `')'` 的字符串，找出最长有效（格式正确且连续）括号子串的长度。

定义 dp[i] 表示以下标 *i* 字符结尾的最长有效括号的长度。

难点在于怎么思考状态转移方程。

想到如果是合法的括号，那么肯定是左右都有，那么是不是可以每次走两步

*s*[*i*]=‘)’ 且 *s*[*i*−1]=‘(’，也就是字符串形如 “……()”，我们可以推出：dp[i]=dp[i−2]+2

**key：** 两种有效的括号类型：（...）（...）（...），另一种为嵌套格式 （（...））

第一种可以 dp[i] = dp[i - 2] + 2，第二种则比较复杂。

考虑第一次遇到 ... ））时，需要找到和右括号匹配的左括号，我们这里可以根据最优子结构得到 dp[i - 1] 代表了前一个右括号之前的有效括号，那么 i - dp[i - 1] - 1 的位置就是和当前右括号匹配的位置，如果这个位置是左括号，那么最长有效长度就可以 + 2，否则就不更新。

![截屏2020-04-17下午4.26.34.png](./6e07ddaac3b703cba03a9ea8438caf1407c4834b7b1e4c8ec648c34f2833a3b9-截屏2020-04-17下午4.26.34.png)

同时还要考虑 ...((...)) 的情况，也就是加上 dp[i - dp[i - 1] - 2]

于是有以下内容：

```java
class Solution {
    public int longestValidParentheses(String s) {
        int n = s.length();
        int[] dp = new int[n]; // 以下标 i 结尾的最长子串长度
        int max = 0;
        for (int i = 1; i < n; i++) {
            if (s.charAt(i - 1) == '(' && s.charAt(i) == ')') {
                // 能够处理 ..()，但是无法处理 (())
                dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2;
            } else if (s.charAt(i - 1) == ')' && s.charAt(i) == ')') {
                // 这里处理 ...((...)) 的情况
                // 判断 s[i] 是否有对应的左括号
                if (i - dp[i - 1] - 1 >= 0 && s.charAt(i - dp[i - 1] - 1) == '(') {
                    // 第一项对应 ((...)) 中的三个点
                    // 最后一项是和匹配的括号的情况连起来，对应 ...(()) 中的三个点
                    // dp[i] = dp[i - 1] + 2 + (i - dp[i - 1] - 2) > 0 ? dp[i - dp[i - 1] - 2] : 0;
                    dp[i] = dp[i - 1] + 2 + ((i - dp[i - 1] - 2) > 0 ? dp[i - dp[i - 1] - 2] : 0);
                }
            }
            max = Math.max(max, dp[i]);
        }
        return max;
    }
}
```

### 栈

如果用栈，那么关键在于对于 ...(...)() 和 ...(()) 这两种情况如何判断长度和判断连续。

对于连续来说，可以过一遍字符串即可，对于长度判断，要点在于，如果对于右括号没有对应的左括号时，说明需要另起炉灶，重新计算最大长度。

```java
class Solution {
    public int longestValidParentheses(String s) {
        int max = 0;
        Deque<Integer> stack = new LinkedList<Integer>();
        stack.push(-1);
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                stack.push(i);
            } else {
                stack.pop();
                if (stack.isEmpty()) {
                    stack.push(i);
                } else {
                    max = Math.max(max, i - stack.peek());
                }
            }
        }
        return max;
    }
}
```



## 寻找两个正序数组的中位数[^3]

中位数定义：将一个集合划分为两个长度相等的子集，其中一个子集中的元素总是大于另一个子集中的元素。

          left_part          |         right_part
    A[0], A[1], ..., A[i-1]  |  A[i], A[i+1], ..., A[m-1]
    B[0], B[1], ..., B[j-1]  |  B[j], B[j+1], ..., B[n-1]

根据中位数的定义，我们需要找到以上的划分（设两个数组总长度为偶数）使得

- $\text{len}(left\_part) = \text{len}(right\_part)$
- $\max(left\_part)=\max(right\_part)$

此时的中位数为：

$$\text{median} = \frac{\max(left\_part)+\min(right\_part)}{2}$$

所以现在的问题关键在于寻找这样一个划分。要寻找这样一个划分需要根据这个划分满足的两个条件：

- 左边元素共有 $i + j$ 个，右边元素共有 $(m-i)+(n-j)$ 个，所以由第一个式子可以得到 $i+j=(m-i)+(n-j)$。变形得到 $i+j=\frac{m+n}{2}$。假设 $m < n$，即 B 数组长于 A 数组，则 $i\in[0,m]$，有 $j = \frac{m+n}{2}-i$ 且 $j \in [0,n]$，所以只要知道 $i$ 的值，那么 $j$ 的值也是确定的。
- 在 $(0, m)$ 中找到 $i$，满足 $A[i-1] \le B[j]$ 且 $A[i] \ge B[j-1]$ 。

注意到第一个条件中，当 $i$ 增大的时候，$j$ 会减小以此来保证左右两部分的元素个数相同。同时 A、B 数组都是单调不递减的，所以一定存在一个最大的 $i$ 满足 $A[i-1] \le B[j]$。（当 $i$ 取 $i+1$ 时 $A[i] > B[j-1]$）

所以问题转化为：找一个最大的 $i$ 使得 $A[i-1] \le B[j]$。

对于这个问题，我们容易枚举 $i$，同时 A、B 都是单调递增的，所以我们还能知道枚举出的 $i$ 是不是满足条件（$A[i-1] \le B[j]$），并从中找出满足条件的最大 $i$ 值即可。

对于两个数组总长度为奇数的情况，可以使得 $j = \lfloor \frac{m+n+1}{2}-i \rfloor$。

代码如下：

```rust
#[warn(dead_code)]
struct Solution;

impl Solution {
    pub fn find_median_sorted_arrays(nums1: Vec<i32>, nums2: Vec<i32>) -> f64 {
        if nums1.len() > nums2.len() {
            return Solution::find_median_sorted_arrays(nums2, nums1);
        }
        // m < n
        let (m, n) = (nums1.len(), nums2.len());
        let mut left = 0;
        let mut right = m;
        let mut pos = 0;
        let mut median1 = 0;
        let mut median2 = 0;
        while left <= right {
            let i = (left + right) / 2;
            let j = (m + n + 1) / 2 - i;
            let nums_im1 = if i == 0 { -0x3f3f3f3f } else { nums1[i - 1] };
            let nums_i = if i == m { 0x3f3f3f3f } else { nums1[i] };
            let nums_jm1 = if j == 0 { -0x3f3f3f3f } else { nums2[j - 1] };
            let nums_j = if j == n { 0x3f3f3f3f } else { nums2[j] };
            if nums_im1 <= nums_j {
                median1 = std::cmp::max(nums_im1, nums_jm1);
                median2 = std::cmp::min(nums_i, nums_j);
                left = i + 1;
            } else {
                right = i - 1;
            }
        }
        if (m + n) & 1 == 0 {
            (median1 + median2) as f64 / 2.0
        } else {
            median1 as f64
        }
    }
}
```



## 三数之和

### 朴素算法

排序之后三重循环，判断三个数之和是否为 $0$，时间复杂度 $O(n^3)$。

排序的目的是为了容易地去除重复数字，因为排序之后只需要判断当前和前一个元素是否相等就可以知道是否是重复数字。

### 排序后双指针

注意到排序之后整个数组是单调非递减的，我们需要 $a+b+c=0$，当固定了 $a$ 和 $b$ 的时候，$c$ 从大到小地判断是否有 $a+b+c=0$ 即可。看似是最外层对应 $a$ 的循环嵌套对应 $b$ 的循环，并在其中加上了 $c$ 递减的循环，但是实际上注意到当 $b$ 与 $c$ 是同一个元素时，如果仍然不满足 $a+b+c=0$，那么 $c$ 继续向左减小就与之前的数字重复了，所以对于每一次 $b$ 中的循环，最多运行 $n$ 次，外边再嵌套 $a$ 的循环，时间复杂度为 $O(n^2)$。

代码如下：

```rust
#[warn(dead_code)]
struct Solution;

impl Solution {
    pub fn three_sum(mut nums: Vec<i32>) -> Vec<Vec<i32>> {
        nums.sort();
        let len = nums.len();
        let mut ans = Vec::new();
        for i in 0..len {
            // 防止取到相同的数字
            if i > 0 && nums[i - 1] == nums[i] {
                continue;
            }
            let mut third = len - 1;
            // 注意这里开始位置是 i+1，目的是为了不与 a 取重
            for j in i + 1..len {
                // 注意这里判定条件是 j > i+1 否则会取不到与 a 相同的数字
                if j > i + 1 && nums[j - 1] == nums[j] {
                    continue;
                }
                while j < third && nums[i] + nums[j] + nums[third] > 0 {
                    third = third - 1;
                }
                if j == third {
                    break;
                }
                if nums[i] + nums[j] + nums[third] == 0 {
                    ans.push(vec![nums[i], nums[j], nums[third]]);
                }
            }
        }
        ans
    }
}
```

## 盛最多水的容器[^4]

$$
area = (right - left) * \min (height[left], height[right])
$$

由上面的公式可以知道，面积由两部分共同决定：

- 宽度
- 高度

所以考虑尽可能地增加宽度和高度。假设左指针指向的数为 $x$，右指针指向的数为 $y$，假设 $x < y$，距离为 $t$，接下来进行具体分析：

1. 水量 $ area = \min(x, y) * t = x * t $，当左指针不变的时候，右指针无论在哪都不会影响容器的水量了，水量是固定的 $x*t$。
2. 所以考虑左指针向右移动，这样才有可能取到更大的水量。
3. 同理左指针指向的数大于右指针指向的数的时候，左移右指针才有可能取到更大的水量。
4. 重复以上步骤就可以得到最大水量。

总时间复杂度为 $O(n)$。

注解：

- 对于双指针问题，两个指针的初始位置不一定都在最左或者最右，要灵活地设置指针位置。

## 最接近三数之和

与「盛最多水的容器」和「三数之和」类似，代码如下：

```rust
#[warn(dead_code)]
struct Solution;

impl Solution {
    pub fn three_sum_closest(mut nums: Vec<i32>, target: i32) -> i32 {
        nums.sort();
        let len = nums.len();
        let mut ans = 0;
        let mut diff = 0x3f3f3f3f;
        for i in 0..len {
            if i > 0 && nums[i] == nums[i - 1] {
                continue;
            }
            let mut j = i + 1;
            let mut k = len - 1;
            while j < k {
                //dbg!((i, j , k));
                let sum = nums[i] + nums[j] + nums[k];
                if sum == target {
                    return sum;
                }
                let tmp = (sum - target).abs();
                if tmp < diff {
                    diff = tmp;
                    ans = sum;
                }
                if sum > target {
                    let mut k0 = k - 1;
                    while j < k0 && nums[k0] == nums[k] {
                        k0 = k0 - 1;
                    }
                    k = k0;
                } else {
                    let mut j0 = j + 1;
                    while j0 < k && nums[j0] == nums[j] {
                        j0 = j0 + 1;
                    }
                    j = j0;
                }
            }
        }
        ans
    }
}
```

# 数据结构

## 合并K个升序链表

使用优先队列即可。

```rust
use std::{cmp::Reverse, collections::BinaryHeap};
impl Solution {
    pub fn merge_k_lists(lists: Vec<Option<Box<ListNode>>>) -> Option<Box<ListNode>> {
        let mut priority_queue = BinaryHeap::new();
        let mut ret = Box::new(ListNode::new(0));
        let mut ptr = &mut ret;
        for list in lists {
            let mut plist = &list;
            while let Some(node) = plist {
                priority_queue.push(Reverse(node.val));
                plist = &node.next;
            }
        }

        while let Some(Reverse(node)) = priority_queue.pop() {
            ptr.next = Some(Box::new(ListNode::new(node)));
            ptr = ptr.next.as_mut().unwrap();
        }
        ret.next
    }
}
```



# 参考

[^1]: https://leetcode-cn.com/problems/longest-palindromic-substring/solution/xiang-xi-tong-su-de-si-lu-fen-xi-duo-jie-fa-bao-gu
[^2]: https://oi-wiki.org/string/manacher/

[^3]: https://leetcode-cn.com/problems/median-of-two-sorted-arrays/solution/xun-zhao-liang-ge-you-xu-shu-zu-de-zhong-wei-s-114/
[^4]: https://leetcode-cn.com/problems/container-with-most-water/solution/sheng-zui-duo-shui-de-rong-qi-by-leetcode-solution/

