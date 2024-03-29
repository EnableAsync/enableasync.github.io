---
title: Rust的一些学习心得
date: 2021-12-08 19:04:50
tags: rust
categories: rust
---

# Rust 标准库 trait

假设有以下变量：

```rust
let t = T::new()
```

## `impl From<U> for T`

如果为 `T` 实现了 `From<U>` 则可以通过 `T::from(U)` 得到 `T`。

例如 `String` 实现了 `From<&str>`，所以 `String` 可以从 `&str` 生成。

```rust
let string = "hello".to_string();
let other_string = String::from("hello");

assert_eq!(string, other_string);
```

`impl Into<U> for T`

如果为 `T` 实现了 `Into<U>` 则可以通过 `t.into()` 消耗自己得到 `U`。

例如 `String` 类型实现了 `Into<Vec<u8>>`。

```rust
fn is_hello<T: Into<Vec<u8>>>(s: T) {
   let bytes = b"hello".to_vec();
   assert_eq!(bytes, s.into());
}

let s = "hello".to_string();
is_hello(s);
```

在实际编程中，用来接收多种类型的参数，如 `Into<String>` 可以同时接收 `String` 和 `&str`。

## `impl AsRef<U> for T`

如果为 `T` 实现了 `AsRef<U>` 则可以通过 `t.as_ref()` 得到 `&U`。

注：

1. 与 `Into<U>` 不同的是，`AsRef<U>` 只是类型转换，`t` 对象本身没有被消耗；
2. `T: AsRef<U>` 中的 `T`，可以接受 资源拥有者（owned）类型，共享引用（shared referrence）类型 ，可变引用（mutable referrence）类型。

例如 `String` 和 `&str` 都实现了 `AsRef<str>`：

```rust
fn is_hello<T: AsRef<str>>(s: T) {
   assert_eq!("hello", s.as_ref());
}

let s = "hello";
is_hello(s);

let s = "hello".to_string();
is_hello(s);
```

## `impl AsMut<U> for T`

如果为 `T` 实现了 `AsRef<U>` 则可以通过 `t.as_mut()` 得到 `&mut U`。

## `impl Borror<U> for T`

如果 `T` 实现了 `Borrow<U>`，那么，`t` 可执行 `.borrow()` 操作，即 `t.borrow()`。操作的结果，我们得到了一个类型为 `&U` 的新引用。

`Borrow` 可以认为是 `AsRef` 的严格版本，它对普适引用操作的前后类型之间附加了一些其它限制。

`Borrow` 的前后类型之间要求必须有内部等价性。不具有这个等价性的两个类型之间，不能实现 `Borrow`。

`AsRef` 更通用，更普遍，覆盖类型更多，是 `Borrow` 的超集。

举例：

```rust
use std::borrow::Borrow;

fn check<T: Borrow<str>>(s: T) {
    assert_eq!("Hello", s.borrow());
}

let s = "Hello".to_string();

check(s);

let s = "Hello";

check(s);
```

## `impl BorrowMut<U> for T`

如果 `T` 实现了 `BorrowMut<U>`，那么，`t` 可执行 `.borrow_mut()` 操作，即 `t.borrow_mut()`。操作的结果我们得到类型为 `&mut U` 的一个可变（mutable）引用。

## `impl ToOwned for T`

`ToOwned` 为 `Clone` 的普适版本。它提供了 `.to_owned()` 方法，用于类型转换。

有些实现了 `Clone` 的类型 `T` 可以从引用状态实例 `&T` 通过 `.clone()` 方法，生成具有所有权的 `T` 的实例。但是它只能由 `&T` 生成 `T`。而对于其它形式的引用，`Clone` 就无能为力了。

而 `ToOwned` trait 能够从任意引用类型实例，生成具有所有权的类型实例。

## `impl Deref for T`

`Deref` 是 `deref` 操作符 `*` 的 trait，比如 `*v`。

一般理解，`*t` 操作，是 `&t` 的反向操作，即试图由资源的引用获取到资源的拷贝（如果资源类型实现了 `Copy`），或所有权（资源类型没有实现 `Copy`）。

Rust 中，本操作符行为可以重载。这也是 Rust 操作符的基本特点。本身没有什么特别的。

### 强制隐式转换（coercion）

`Deref` 神奇的地方并不在本身 `解引` 这个意义上，Rust 的设计者在它之上附加了一个特性：`强制隐式转换`，这才是它神奇之处。

这种隐式转换的规则为：

一个类型为 `T` 的对象 `t`，如果 `T: Deref<Target=U>`，那么，相关 `t` 的某个智能指针或引用（比如 `&foo`）在应用的时候会自动转换成 `&U`。

粗看这条规则，貌似有点类似于 `AsRef`，而跟 `解引` 似乎风马牛不相及。实际里面有些玄妙之处。

Rust 编译器会在做 `*v` 操作的时候，自动先把 `v` 做引用归一化操作，即转换成内部通用引用的形式 `&v`，整个表达式就变成 `*&v`。这里面有两种情况：

1. 把其它类型的指针（比如在库中定义的，`Box`, `Rc`, `Arc`, `Cow` 等），转成内部标准形式 `&v`；
2. 把多重 `&` （比如：`&&&&&&&v`），简化成 `&v`（通过插入足够数量的 `*` 进行解引）。

所以，它实际上在解引用之前做了一个引用的归一化操作。

为什么要转呢？ 因为编译器设计的能力是，只能够对 `&v` 这种引用进行解引用。其它形式的它不认识，所以要做引用归一化操作。

使用引用进行过渡也是为了能够防止不必要的拷贝。

下面举一些例子：

```rust
fn foo(s: &str) {
    // borrow a string for a second
}

// String implements Deref<Target=str>
let owned = "Hello".to_string();

// therefore, this works:
foo(&owned);
```

因为 `String` 实现了 `Deref<Target=str>`。

```rust
use std::rc::Rc;

fn foo(s: &str) {
    // borrow a string for a second
}

// String implements Deref<Target=str>
let owned = "Hello".to_string();
let counted = Rc::new(owned);

// therefore, this works:
foo(&counted);
```

因为 `Rc<T>` 实现了 `Deref<Target=T>`。

```rust
fn foo(s: &[i32]) {
    // borrow a slice for a second
}

// Vec<T> implements Deref<Target=[T]>
let owned = vec![1, 2, 3];

foo(&owned);
```

因为 `Vec<T>` 实现了 `Deref<Target=[T]>`。

```rust
struct Foo;

impl Foo {
    fn foo(&self) { println!("Foo"); }
}

let f = &&Foo;

f.foo();
(&f).foo();
(&&f).foo();
(&&&&&&&&f).foo();
```

上面那几种函数的调用，效果是一样的。

`coercion` 的设计，是 Rust 中仅有的类型隐式转换，设计它的目的，是为了简化程序的书写，让代码不至于过于繁琐。把人从无尽的类型细节中解脱出来，让书写 Rust 代码变成一件快乐的事情。

## `Cow`

`Clone-on-write`，即写时克隆。本质上是一个智能指针。

它有两个可选值：

- `Borrowed`，用于包裹对象的引用（通用引用）；
- `Owned`，用于包裹对象的所有者；

`Cow` 提供

1. 对此对象的不可变访问（比如可直接调用此对象原有的不可变方法）；
2. 如果遇到需要修改此对象，或者需要获得此对象的所有权的情况，`Cow` 提供方法做克隆处理，并避免多次重复克隆。

`Cow` 的设计目的是提高性能（减少复制）同时增加灵活性，因为大部分情况下，业务场景都是读多写少。利用 `Cow`，可以用统一，规范的形式实现，需要写的时候才做一次对象复制。这样就可能会大大减少复制的次数。

它有以下几个要点需要掌握：

1. `Cow<T>` 能直接调用 `T` 的不可变方法，因为 `Cow` 这个枚举，实现了 `Deref`；
2. 在需要写 `T`的时候，可以使用 `.to_mut()` 方法得到一个具有所有权的值的可变借用；
   1. 注意，调用 `.to_mut()` 不一定会产生克隆；
   2. 在已经具有所有权的情况下，调用 `.to_mut()` 有效，但是不会产生新的克隆；
   3. 多次调用 `.to_mut()` 只会产生一次克隆。
3. 在需要写 `T` 的时候，可以使用 `.into_owned()` 创建新的拥有所有权的对象，这个过程往往意味着内存拷贝并创建新对象；
   1. 如果之前 `Cow` 中的值是借用状态，调用此操作将执行克隆；
   2. 本方法，参数是`self`类型，它会“吃掉”原先的那个对象，调用之后原先的对象的生命周期就截止了，在 `Cow` 上不能调用多次；

### 例子

`.to_mut()` 举例

```rust
use std::borrow::Cow;

let mut cow: Cow<[_]> = Cow::Owned(vec![1, 2, 3]);

let hello = cow.to_mut();

assert_eq!(hello, &[1, 2, 3]);
```

`.into_owned()` 举例

```rust
use std::borrow::Cow;

let cow: Cow<[_]> = Cow::Owned(vec![1, 2, 3]);

let hello = cow.into_owned();

assert_eq!(vec![1, 2, 3], hello);
```

综合举例

```rust
use std::borrow::Cow;

fn abs_all(input: &mut Cow<[i32]>) {
    for i in 0..input.len() {
        let v = input[i];
        if v < 0 {
            // clones into a vector the first time (if not already owned)
            input.to_mut()[i] = -v;
        }
    }
}
```

### 更多的例子

题目：写一个函数，过滤掉输入的字符串中的所有空格字符，并返回过滤后的字符串。

对这个简单的问题，不用思考，我们都可以很快写出代码：

```rust
fn remove_spaces(input: &str) -> String {
   let mut buf = String::with_capacity(input.len());

   for c in input.chars() {
      if c != ' ' {
         buf.push(c);
      }
   }

   buf
}
```

设计函数输入参数的时候，我们会停顿一下，这里，用 `&str` 好呢，还是 `String` 好呢？思考一番，从性能上考虑，有如下结论：

1. 如果使用 `String` 则外部在调用此函数的时候，
   1. 如果外部的字符串是 `&str`，那么，它需要做一次克隆，才能调用此函数；
   2. 如果外部的字符串是 `String`，那么，它不需要做克隆，就可以调用此函数。但是，一旦调用后，外部那个字符串的所有权就被 `move` 到此函数中了，外部的后续代码将无法再使用原字符串。
2. 如果使用 `&str`，则不存在上述两个问题。但可能会遇到生命周期的问题，需要注意。

继续分析上面的例子，我们发现，在函数体内，做了一次新字符串对象的生成和拷贝。

让我们来仔细分析一下业务需求。最坏的情况下，如果字符串中没有空白字符，那最好是直接原样返回。这种情况做这样一次对象的拷贝，完全就是浪费了。

于是我们心想改进这个算法。很快，又遇到了另一个问题，返回值是 `String` 的嘛，我不论怎样，要把 `&str` 转换成 `String` 返回，始终都要经历一次复制。于是我们快要放弃了。

好吧，`Cow` 君这时出马了。写出了如下代码：

```rust
use std::borrow::Cow;

fn remove_spaces<'a>(input: &'a str) -> Cow<'a, str> {
    if input.contains(' ') {
        let mut buf = String::with_capacity(input.len());

        for c in input.chars() {
            if c != ' ' {
                buf.push(c);
            }
        }

        return Cow::Owned(buf);
    }

    return Cow::Borrowed(input);
}
```

完美解决了业务逻辑与返回值类型冲突的问题。本例可细细品味。

外部程序，拿到这个 `Cow` 返回值后，按照我们上文描述的 `Cow` 的特性使用就好了。

## `Send` 和 `Sync`

`std::marker` 模块中，有两个 trait：`Send` 和 `Sync`，它们与多线程安全相关。

标记为 `marker trait` 的 trait，它实际就是一种约定，没有方法的定义，也没有关联元素（associated items）。仅仅是一种约定，实现了它的类型必须满足这种约定。一种类型是否加上这种约定，要么是编译器的行为，要么是人工手动的行为。

`Send` 和 `Sync` 在大部分情况下（针对 Rust 的基础类型和 std 中的大部分类型），会由编译器自动推导出来。对于不能由编译器自动推导出来的类型，要使它们具有 `Send` 或 `Sync` 的约定，可以由人手动实现。实现的时候，必须使用 `unsafe` 前缀，因为 Rust 默认不信任程序员，由程序员自己控制的东西，统统标记为 `unsafe`，出了问题（比如，把不是线程安全的对象加上 `Sync` 约定）由程序员自行负责。

它们的定义如下：

如果 `T: Send`，那么将 `T` 传到另一个线程中时（按值传送），不会导致数据竞争或其它不安全情况。

1. `Send` 是对象可以安全发送到另一个执行体中；
2. `Send` 使被发送对象可以和产生它的线程解耦，防止原线程将此资源释放后，在目标线程中使用出错（use after free）。

如果 `T: Sync`，那么将 `&T` 传到另一个线程中时，不会导致数据竞争或其它不安全情况。

1. `Sync` 是可以被同时多个执行体访问而不出错；
2. `Sync` 防止的是竞争；

推论：

1. `T: Sync` 意味着 `&T: Send`；
2. `Sync + Copy = Send`；
3. 当 `T: Send` 时，可推导出 `&mut T: Send`；
4. 当 `T: Sync` 时，可推导出 `&mut T: Sync`；
5. 当 `&mut T: Send` 时，不能推导出 `T: Send`；

（注：`T`, `&T`, `&mut T`，`Box<T>` 等都是不同的类型）

具体的类型：

1. 原始类型（比如： u8, f64），都是 `Sync`，都是 `Copy`，因此都是 `Send`；
2. 只包含原始类型的复合类型，都是 `Sync`，都是 `Copy`，因此都是 `Send`；
3. 当 `T: Sync`，`Box<T>`, `Vec<T>` 等集合类型是 `Sync`；
4. 具有内部可变性的的指针，不是 `Sync` 的，比如 `Cell`, `RefCell`, `UnsafeCell`；
5. `Rc` 不是 `Sync`。因为只要一做 `&Rc<T>` 操作，就会克隆一个新引用，它会以非原子性的方式修改引用计数，所以是不安全的；
6. 被 `Mutex` 和 `RWLock` 锁住的类型 `T: Send`，是 `Sync` 的；
7. 原始指针（`*mut`, `*const`）既不是 `Send` 也不是 `Sync`；

Rust 正是通过这两大武器：`所有权和生命周期` + `Send 和 Sync`（本质上为类型系统）来为并发编程提供了安全可靠的基础设施。使得程序员可以放心在其上构建稳健的并发模型。这也正是 Rust 的核心设计观的体现：内核只提供最基础的原语，真正的实现能分离出去就分离出去。并发也是如此。

# 参考文献

1. https://github.com/rustcc/RustPrimer

