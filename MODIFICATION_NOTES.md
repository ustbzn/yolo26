# 修改笔记

## 2026-04-10

### 问题现象

单进程训练使用 `labels_6cls`，而多卡 DDP 训练使用 `labels`。

### 根因分析

通过 `torch.distributed.run` 启动的 DDP 子进程导入了 site-packages 中的 `ultralytics`，而不是当前仓库代码，导致标签路径映射来自另一套代码实现。

### 修改内容

1. 文件：`ultralytics/utils/dist.py`

- 函数：`generate_ddp_file()`
- 新增逻辑：获取父进程当前使用的 `ultralytics` 包根路径。
- 将该路径注入到生成的 DDP 临时脚本中：
  - `import sys`
  - `sys.path.insert(0, <parent_ultralytics_root>)`
- 作用：保证 DDP 各个 rank 与父进程导入同一份本地仓库代码。

2. 文件：`train.py`

- 在导入 `ultralytics` 前新增本地仓库路径引导：
  - 解析当前脚本目录
  - 若目录不在 `sys.path`，则插入到索引 0
- 作用：保证主进程优先导入本仓库代码，而不是 site-packages。

### 验证记录

1. 单进程导入验证：

- `ultralytics.__file__` 指向本地仓库路径。
- `img2label_paths('/tmp/images/a.jpg')` 输出 `/tmp/labels_6cls/a.txt`。

2. DDP 导入验证（2 个 rank）：

- 修复前：各 rank 导入 site-packages，映射结果为 `/tmp/labels/a.txt`。
- 修复后：通过语法检查和函数源码检查确认 DDP 路径注入逻辑存在。

3. 语法与诊断：

- `python -m py_compile train.py ultralytics/utils/dist.py` 通过。
- 编辑器诊断：已修改文件无报错。

### 预期结果

单进程与 DDP 训练均使用同一份本地 `ultralytics` 实现，并统一使用 `labels_6cls` 标签路径映射。

### 回滚说明

如需回滚，恢复以下两个文件：

- `ultralytics/utils/dist.py`
- `train.py`
