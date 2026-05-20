# PINN-HJB-2-CORRECT
PINN-HJB-2

基于 PINN-HJB 方法的海洋平台系泊失效控制项目。

## 当前状态

**v1 分支**

- ✅ 基础功能已跑通
- ⚠️ 外推数据存在错误，导致力输出偏小
- 🔄 后续待修复外推问题
- ❌完全错误，因为代理模型根本拟合不好，loss值太大，需要重新设计训练函数
## 文件说明

| 文件  | 说明  |
| --- | --- |
| `pinn_hjb_controller.py` | PINN-HJB 控制器核心 |
| `pinn_hjb_aqwa_integration.py` | AQWA 集成层 |
| `pinn_hjb_aqwa_online.py` | 在线控制部署 |
| `surrogate_trainer.py` | 代理模型训练 |
| `calibrate_r_for_u_scale.py` | 参数校准 |
| `requirements.txt` | Python 依赖 |

## 环境

- Python 3.x
- PyTorch
- AQWA（水动力分析）

## 作者

111-kong
