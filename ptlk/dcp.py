#!/usr/bin/env python
# -*- coding: utf-8 -*-
try:
    from .dcp.model import DCP as _DCP
except ImportError:
    # 如果找不到 dcp.model 模块，则定义一个空基类以避免导入失败
    class _DCP:
        def __init__(self, args):
            # 空基类，无实际功能
            pass

class DCP(_DCP):
    """
    基于 DCP 仓库，实现与 PointNetLK 接口一致的包装
    """
    def __init__(self, args):
        super(DCP, self).__init__(args) 