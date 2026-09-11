"""知识库构建工具链（离线批处理）。

分工：
- extract.py   异构格式抽取（EPUB / PDF / MOBI-AZW / TXT）
- normalize.py 中医古籍文本规范化
- taxonomy.py  学派与文档角色归类
- pipeline.py  编排、断点续传、产物清单
- evaluate.py  检索质量评测（golden set）
"""
