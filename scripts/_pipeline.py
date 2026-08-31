# -*- coding: utf-8 -*-
"""数字前缀脚本的导入桥（辅助模块，不单独运行）。

脚本按流水线运行顺序命名为 01_ ~ 16_，但 Python 不允许 import 以数字开头的模块名，
而 06_eval_sft.py（打分口径）/ 07_build_dpo_data.py（种子加载与参数表）/
12_run_final_eval.py（竞技场评审）里的函数被多个下游脚本复用。

本模块把这三个文件注册成无前缀别名，使下游脚本里的
    import _pipeline
    from eval_sft import score_answer
继续生效——所有版本对比因此仍共用同一套打分函数（同一把尺子）。
"""
import importlib.util
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))

# 别名 -> 实际文件名；顺序即加载顺序（后者依赖前者）
_ALIASES = {
    "eval_sft": "06_eval_sft.py",
    "build_dpo_data": "07_build_dpo_data.py",
    "build_arena_dashboard": "13_build_arena_dashboard.py",
    "run_final_eval": "12_run_final_eval.py",
}


def load(alias: str):
    """按别名加载带数字前缀的脚本模块，已加载则直接复用。"""
    if alias in sys.modules:
        return sys.modules[alias]
    path = os.path.join(_DIR, _ALIASES[alias])
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module      # 先注册再执行，允许模块间相互引用
    spec.loader.exec_module(module)
    return module


for _alias in _ALIASES:
    load(_alias)
