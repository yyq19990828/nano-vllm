# 概念速查表

学习过程中遇到的零碎知识点, 按字母排序, 方便快速查阅。
归属明确的知识点应写入对应主题笔记, 这里只放跨主题或基础概念。

---

### chat_template
Qwen3 用 Jinja2 模板定义对话格式。`apply_chat_template()` 把消息列表渲染成模型需要的文本,
如 `<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n`。
模板里大量 if/for 分支处理 tools、think、多轮对话等场景, 没命中的分支不产生输出。

### Jinja2
Python 模板引擎, 语法: `{{ 变量 }}`, `{% if %}`, `{% for %}`。
原本用于 Web 渲染 HTML, HuggingFace 用它定义 chat_template。不是独立语言。

