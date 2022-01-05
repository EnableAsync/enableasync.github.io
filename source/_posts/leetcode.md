---
title: 算法整理
date: 2022-01-04 19:21:50
tags: data structure, algorithm
image: https://dingxuewen.com/leetcode-js-leviding/leetcode.png
mathjax: true
---

# 字符串

## 最长公共子串

状态转移方程如下：
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


# 参考
[^1]: https://leetcode-cn.com/problems/longest-palindromic-substring/solution/xiang-xi-tong-su-de-si-lu-fen-xi-duo-jie-fa-bao-gu
[^2]: https://oi-wiki.org/string/manacher/
