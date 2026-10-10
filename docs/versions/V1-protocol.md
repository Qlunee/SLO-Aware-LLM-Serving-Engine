# V1规则归档与校验

长段终端粘贴出现未闭合字符串时，Python会在执行前报SyntaxError，冻结逻辑尚未执行。此前由`mktemp`创建的空目录可以保留，新的入口会忽略它。

使用独立脚本，不再向终端粘贴整段Python：

```bash
cd /home/lq/project/SLO-Aware-LLM-Serving-Engine
source scripts/env_qwen3_4b.sh
"$V0_PYTHON" -B scripts/freeze_v1_protocol.py freeze
"$V0_PYTHON" -B scripts/freeze_v1_protocol.py verify --latest
```

新建`results/raw/v1-protocol.*`，保存policy、sources、evidence及manifest四个JSON文件。先检查全部引用证据，再归档规则；源文件集合包含Python、Shell和配置，包括冻结工具本身。校验同时检测已记录文件的变化、新增或删除，以及证据manifest引用的二进制产物变化。

例外的历史来源：早期手动`v1-first-divergence.005ytr8a`只有report.json与step11.pt，没有实验时manifest。本次直接记录这两个已有文件的当前哈希，不向历史目录补造manifest，也不声称恢复了实验时的完整性证明。

`--latest`选择最近完成（已有manifest）的目录，输出协议目录及policy SHA256。后续实验建议用打印的具体路径执行`verify --dir /absolute/path/to/protocol`，明确绑定同一份协议。脚本不需要`V1_PROTOCOL_DIR`环境变量。

冻结只归档，不自动运行或豁免验收。沿用已有阈值，近并列不自动通过，旧失败保持失败。语义、数值兼容性及质量分别记录；原端到端数值失败仍阻止全项目PASS。更改范围或阈值须建立新规则版本，说明原因，并用新实验验证。旧产物不覆盖。

该工具不调用GPU、不改变推理代码，不向现有测试注入规则。工具回归测试：

```bash
"$V0_PYTHON" -B -m unittest discover -s tests -p test_freeze_protocol.py -v
```
