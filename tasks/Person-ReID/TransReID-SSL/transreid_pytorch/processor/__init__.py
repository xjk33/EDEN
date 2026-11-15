# processor.py（原始 DenoiseRep 训练循环）依赖 denoiserep 包——那是原 DenoiseRep
# 项目的私有包（仓库根 ../../denoiserep_op/），不在 EDEN 代码树内。EDEN 训练链路
# （train_eden.py -> processor.eden_processor）不需要它；denoiserep 缺失时降级为
# None，不影响 from processor.eden_processor import do_train_eden。
try:
    from .processor import do_train, do_inference
except ModuleNotFoundError:
    do_train = do_inference = None
