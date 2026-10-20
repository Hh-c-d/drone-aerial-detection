# Drone Aerial Object Detection
YOLO-based detection for small objects in drone footage (vehicles, people, boats).

> Status: project scaffold. All metrics below are placeholders — fill in only real results from your own runs.

## Demo
[效果图 / GIF 待放]

## Problem
航拍巡检、安防、搜救场景需要检测车辆、人员、船只。航拍图目标小、密度高、拍摄角度特殊，普通场景训练的模型直接套用效果差。

## Dataset
- VisDrone（公开航拍数据集），先取子集跑通流程
- 图片数：[待填]
- 类别：[待填]
- 划分方式：[待填]

## Method
- 模型：Ultralytics YOLO（YOLO11 / YOLO26）
- 训练参数：[待填]
- 关键技巧：切片推理（SAHI 思路），与整图推理做对比

## Results
| 模型 | 方式 | mAP50 | mAP50-95 |
| --- | --- | --- | --- |
| [待填] | 整图推理 | [待填真实结果] | [待填] |
| [待填] | 切片推理 | [待填真实结果] | [待填] |

失败案例分析：[待填]

## Quick Start
pip install ultralytics
yolo predict model=best.pt source=your_image.jpg

## Deliverables
- best.pt 训练权重
- 推理脚本
- 训练配置
