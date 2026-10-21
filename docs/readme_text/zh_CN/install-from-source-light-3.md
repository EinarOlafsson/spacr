` --dir ` 选择克隆路径；` --branch ` 选择分支或标签；` --no-install ` 仅下载、不安装；` --help ` 列出选项。` packaging/source_install_excludes.txt ` 列出了所有被跳过的路径。

nightly 的完整克隆下载了 9.25 GiB。向浅克隆添加 ` --filter=blob:none ` 不会缩小检出文件：其 Git 对象存储仍占用 2032 MB。Git 不报告按需延迟获取的对象，因此没有测得下载总量。nightly 的版本控制文件检出后占用 720 MB（2026-10-21 测得），不含 Git 历史。下载大小和耗时随分支而变化。
