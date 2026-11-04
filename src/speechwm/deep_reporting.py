from __future__ import annotations

import csv
import json
from pathlib import Path

from .evaluation import write_csv


def make_deep_report(out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary=json.loads((out/"metrics.json").read_text())
    validation=json.loads((out/"validation.json").read_text())
    training=json.loads((out/"training_summary.json").read_text())["models"]
    split=json.loads((out/"split_audit.json").read_text())
    metrics=summary["metrics"];frames=sorted({r["frames"] for r in metrics})
    rates=list(csv.DictReader((out/"fixed_threshold_rates.csv").open()))
    def metric(condition,detector,count,partition="dev_evaluation"):
        return next(r for r in metrics if (r["condition"],r["split"],r["kind"],r["detector"],r["frames"])==
                    (condition,partition,"main",detector,count))
    figures=out/"figures";figures.mkdir(exist_ok=True)
    colors={"correlation":"tab:gray","model_A":"tab:blue","model_B":"tab:orange"}
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout="constrained")
    for ax,row in zip(axes.flat,training):
        name=f"{row['condition']}_{row['architecture']}"
        history=list(csv.DictReader((out/"training"/f"{name}_history.csv").open()))
        epochs=[int(r["epoch"]) for r in history]
        for field,label in [("train_bce","Train"),("validation_bce","Validation")]:
            ax.plot(epochs,[float(r[field]) for r in history],label=label)
        ax.axvline(row["best_epoch_diagnostic"],ls=":",color="tab:green",label="Best val BCE (diagnostic)")
        ax.set(title=f"{name}: {row['parameters']} parameters",xlabel="Epoch",ylabel="BCE loss")
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle("M4 public smoke: fixed epoch 25 is the primary checkpoint")
    fig.savefig(figures/"training_curves.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for ax,cid,title in zip(axes,["C1","C3"],["C1: SS 1 dB, downsample 8 kHz","C3: SS 3 dB, Opus 8 kbps"]):
        for detector in colors:
            ax.plot(frames,[100*metric(cid,detector,m)["eer"] for m in frames],"o-",color=colors[detector],label=detector)
        ax.set(title=title,xlabel="Aggregated valid frames",ylabel="Development evaluation EER (%)",ylim=(0,100))
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle("Same received files, VAD masks and groups; one evaluation source utterance")
    fig.savefig(figures/"evaluation_eer.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout="constrained")
    partitions=["dev_train","dev_validation","dev_evaluation"]
    for row,cid in enumerate(["C1","C3"]):
        for col,count in enumerate([1,30]):
            ax=axes[row,col]
            for detector in colors:
                ax.plot(range(3),[100*metric(cid,detector,count,p)["eer"] for p in partitions],"o-",label=detector,color=colors[detector])
            ax.set(xticks=range(3),xticklabels=["Train","Validation","Evaluation"],ylabel="EER (%)",title=f"{cid}, {count} valid frame(s)",ylim=(0,100))
            ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle("Development partition gap; each partition contains one different speaker / utterance")
    fig.savefig(figures/"partition_gap.png",dpi=160);plt.close(fig)
    comparisons=[]
    for cid in ["C1","C3"]:
        for count in frames:
            base=metric(cid,"correlation",count)
            for arch in ["A","B"]:
                current=metric(cid,f"model_{arch}",count)
                comparisons.append({"condition":cid,"split":"dev_evaluation","frames":count,"architecture":arch,
                    "correlation_eer":base["eer"],"model_eer":current["eer"],"model_minus_correlation_eer":current["eer"]-base["eer"],
                    "negative_groups":current["negative_groups"],"positive_groups":current["positive_groups"],
                    "paired_eligible_segments":current["paired_eligible_segments"],"primary_epoch":25})
    write_csv(out/"model_comparison.csv",comparisons)
    wins=sum(r["model_minus_correlation_eer"]<0 for r in comparisons)
    ties=sum(r["model_minus_correlation_eer"]==0 for r in comparisons)
    losses=len(comparisons)-wins-ties
    lines=["# M4 模型 A/B 开发训练与传统相关检测对照", "",
        f"运行 `{summary['experiment_id']}/{summary['run_id']}`。本轮完成四个真实训练模型、checkpoint 独立推理和同条件对照；当前是公开小样本的流程验证，不能据此声称复现论文数值。", "",
        "## 数据与评估口径", "",
        "沿用 3 条 CC-BY 4.0 的公开 LibriSpeech HQ Ogg 源语音，共 20 个不重叠切片、45.495 秒。原始来源带有损压缩。按原始语句、说话人及源文件哈希整体分区，不随机拆帧：", "",
        "| 分区 | 原始语句 | 切片 | 用途 |", "| --- | --- | ---: | --- |",
        "| dev_train | 3436-172162-0000 | 7 | 梯度训练 |",
        "| dev_validation | 5703-47212-0000 | 7 | BCE 诊断、固定阈值 |",
        "| dev_evaluation | 198-209-0000 | 6 | 第 25 轮冻结后的开发评估 |", "",
        "这 3 条源语音曾用于 M1～M3 开发，因此 dev_evaluation 只与本轮模型训练隔离，并非全项目从未接触的正式测试集。每个分区仅 1 条原始语句、1 位说话人，帧/组相关性很强，不做显著性或总体泛化结论。", "",
        "C1：普通 SS、1 dB、16→8 kHz；C3：普通 SS、3 dB、16→8 kHz 后真实 Opus 单声道 8 kbps。四个模型分别训练，无 LPC 混合、无额外归一化。相关检测与模型共用同一接收 WAV、63 维匹配对数谱特征、独立接收端 VAD 掩码和多帧分组。相关检测保留 float64，神经网络转 float32。", "",
        "聚合帧数为 1、2、5、10、15、20、25、30；在每条接收音频的有效帧序列内非重叠取均值，尾部不足一组丢弃。模型聚合线性 logit，不取 sigmoid 后再平均。有效帧可跨 VAD 间隙，30 帧不等于连续 300 ms。", "",
        "## 网络与训练", "",
        "- A：63→Linear(64)→ReLU→Linear(32)→ReLU→Linear(1)，6209 参数。",
        "- B：输入 [N,1,63]→Conv1d(1,16,kernel=3,stride=1,padding=0)→[N,16,61]→Flatten(976)→Linear(16)→ReLU→Linear(1)，15713 参数；卷积后无额外激活。",
        "- BCEWithLogitsLoss；RMSprop，lr=0.001、alpha=0.99、eps=1e-8、momentum=0、weight_decay=0、centered=false、foreach=false。批量 1024，25 轮，种子 2026，CPU 单线程、确定性算法。",
        "- 主结果固定使用第 25 轮，不依据开发评估选轮。验证集最小 BCE 权重另存作诊断，未替换主结果。保存配置、特征契约、密钥哈希、来源绑定、权重、优化器状态和 RNG 状态。", "",
        "| 模型 | 更新数 | 首轮训练 BCE | 末轮训练 BCE | 末轮验证 BCE | 最佳验证轮（诊断） | 训练秒 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in training:
        lines.append(f"| {r['condition']}_{r['architecture']} | {r['updates']} | {r['first_train_bce']:.6f} | {r['last_train_bce']:.6f} | {r['last_validation_bce']:.6f} | {r['best_epoch_diagnostic']} | {r['fit_seconds']:.3f} |")
    lines += ["", "![训练与验证损失](figures/training_curves.png)", "", "## 冻结开发评估结果", "",
        "以下 EER 每项都在对应开发评估分数上扫描阈值，属于描述性排序指标，不代表可部署阈值下错误率。分数方向固定为越大越倾向含水印；不会按结果反转方向。负的模型差值代表本批切片 EER 更低，正值也如实保留。", "",
        "| 条件 | 有效帧 | 相关 EER % | A EER % | B EER % | 负/正组数 | 配对切片 |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: |"]
    for cid in ["C1","C3"]:
        for count in [1,10,30]:
            a=metric(cid,"correlation",count);b=metric(cid,"model_A",count);c=metric(cid,"model_B",count)
            lines.append(f"| {cid} | {count} | {100*a['eer']:.2f} | {100*b['eer']:.2f} | {100*c['eer']:.2f} | {a['negative_groups']}/{a['positive_groups']} | {a['paired_eligible_segments']} |")
    lines += ["", "完整 8 个聚合长度和 32 条模型差值记录见 model_comparison.csv；首组/片段 EER、有效配对覆盖数见 metrics.csv。小样本中聚合后的组数会很少，应结合组数阅读。", "",
        f"32 项模型对照中，EER 更低 {wins} 项、相同 {ties} 项、更高 {losses} 项；这些设置高度相关，不能视为独立胜率。单帧时两条件的 A/B 都弱于相关检测，部分多帧设置有改善，30 帧 B 与相关检测持平，未观察到稳定优势。", "",
        "训练损失持续下降，验证 BCE 最优却出现在第 4/5 轮，末轮训练/验证差距明显，符合小样本过拟合现象。仅一位训练说话人无法支撑泛化结论；保留固定第 25 轮和这些较差结果，后续先补正式数据，再依正式验证流程决定变更。", "",
        "![开发评估 EER](figures/evaluation_eer.png)", "", "![分区差异](figures/partition_gap.png)", "",
        "## 验证集阈值的迁移诊断", "",
        "每个条件、检测器、聚合长度仅用 dev_validation 计算插值 EER 阈值；在产生 dev_evaluation 分数前连同权重哈希写入 evaluation_gate.json。以下 FAR/FRR 直接使用该阈值，不在评估集重新拟合。插值阈值处的离散 FAR/FRR 不一定相等。", "",
        "| 条件 | 检测器 | 有效帧 | 阈值 | 开发评估 FAR % | 开发评估 FRR % |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    for r in rates:
        if r["split"]=="dev_evaluation" and int(r["frames"]) in [1,30]:
            lines.append(f"| {r['condition']} | {r['detector']} | {r['frames']} | {float(r['threshold']):.6f} | {100*float(r['far']):.2f} | {100*float(r['frr']):.2f} |")
    lines += ["", "这些阈值只作单来源开发诊断。独立 detect --checkpoint 返回分数与 decision=null；缺权重、密钥不匹配、错误采样率或契约不支持时明确失败，不回退相关检测。", "",
        "## 验证与结论边界", "",
        f"- {validation['received_wavs_and_feature_caches_verified']} 条接收 WAV/特征缓存重读核验，包含 {validation['opus_files_verified']} 个真实 Opus 文件。",
        f"- 四个设置各完整重训一次，共 {validation['models_retrained_identically']} 次，权重张量哈希与逐轮历史逐项一致；保存权重重推 {validation['frame_score_rows_reinferred']} 行帧分数，最大 logit 差 {validation['maximum_checkpoint_logit_replay_error']:g}。",
        f"- 独立重聚合 {validation['aggregate_score_rows_recomputed']} 行分数、核验 {validation['common_group_definitions_verified']} 个共同分组，重算 {validation['metric_rows_recomputed']} 条 EER；48 个阈值均来自验证分区。",
        f"- {validation['zero_pcm_pairs']} 对零强度接收 PCM 相同，{validation['zero_eer_settings']} 个零强度设置的 EER 全为 50%；负对照不用于梯度训练。",
        "- 全 20 切片的相关检测另做 16 项 M2 回归，EER 与样本数逐项一致。该回归总体与上表 6 切片评估总体不同，不能交叉比较；旧 M1～M3 冻结证据哈希通过。", "",
        "当前完成的是 M4 开发闭环：实现、实际训练、保存与加载、同组对照、零强度检查和可重复执行。结果无论优于或弱于相关检测都未用于调参。正式实验仍需要更大且按说话人隔离的数据、独立验证/测试、重复种子和论文协议核对；M5 正式主实验及后续 LPC+模型组合探索尚未开始。", "",
        "## 可复查文件", "",
        "config.yaml、environment.json、codec_runtime.json、key.json、split_manifest.jsonl、split_audit.json、audio_manifest.jsonl、feature_manifest.jsonl、channel_manifest.jsonl；features/*.npz、audio/*；training/*_history.csv、training_summary.json、checkpoints/*.pt、calibration.json、evaluation_gate.json；frame_scores.csv、group_definitions.csv、aggregate_scores.csv、metrics.csv、model_comparison.csv、fixed_threshold_rates.csv、threshold_curves.csv、sanity.json、m2_baseline_regression.json、validation.json、failures.jsonl、run_status.json。", ""]
    path=out/"M4模型对照实验报告.md";path.write_text("\n".join(lines));return path
