from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from .main_evaluation import seed_summary
from .utils import read_jsonl


def make_main_report(out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary=json.loads((out/"metrics.json").read_text());validation=json.loads((out/"validation.json").read_text())
    selection=summary["quality_selection"];metrics=summary["metrics"];seeds=seed_summary(metrics)
    quality=read_jsonl(out/"quality_scores.jsonl");training=json.loads((out/"training_summary.json").read_text())["models"]
    figures=out/"figures";figures.mkdir(exist_ok=True)
    test_quality=[r for r in quality if r["split"]=="test"]
    deltas=sorted({r["delta_db"] for r in test_quality})
    def q(method,delta,field="pesq_nb",split="test",gender=None):
        return np.asarray([r[field] for r in quality if r["method"]==method and r["delta_db"]==delta and r["split"]==split and (gender is None or r["gender"]==gender) and r[field] is not None])
    def statistic(cid,detector,frames):return next(r for r in seeds if (r["condition"],r["detector"],r["frames"])==(cid,detector,frames))
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for ax,field in zip(axes,["pesq_nb","pesq_wb"]):
        for method in ["SS","SS_LPC"]:
            arrays=[q(method,d,field) for d in deltas]
            ax.errorbar(deltas,[x.mean() for x in arrays],yerr=[x.std(ddof=1) for x in arrays],fmt="o-",capsize=3,label=method)
        ax.set(xlabel="Nominal strength (dB)",ylabel=field.upper().replace("_"," "),title="Mean +/- sample SD across 200 test utterances")
        ax.grid(alpha=.2);ax.legend()
    fig.suptitle("E02: pre-channel embedding quality, frozen LibriSpeech subset")
    fig.savefig(figures/"test_quality.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for ax,gender in zip(axes,["F","M"]):
        for method in ["SS","SS_LPC"]:ax.plot(deltas,[q(method,d,gender=gender).mean() for d in deltas],"o-",label=method)
        ax.set(title=f"Official metadata {gender}: 100 utterances, 20 speakers",xlabel="Nominal strength (dB)",ylabel="Mean pre-channel NB PESQ")
        ax.grid(alpha=.2);ax.legend()
    fig.savefig(figures/"quality_by_gender.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for ax,noise in zip(axes,["noise20","noise15"]):
        for method in ["SS","SS_LPC"]:
            values=[statistic(f"E03_{noise}_{method}","correlation",m) for m in [1,10,30]]
            ax.plot([1,10,30],[100*r["eer_mean"] for r in values],"o-",label=method)
        ax.set(title=f"{noise}, strengths selected on validation",xlabel="Aggregated valid frames",ylabel="Test EER (%)",ylim=(0,100))
        ax.grid(alpha=.2);ax.legend()
    fig.suptitle("E03: same global validation-selected strengths applied to independent test speakers")
    fig.savefig(figures/"lpc_noise_test.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    frame_counts=[1,2,5,10,15,20,25,30]
    for ax,cid,title in zip(axes,["C1","C3"],["E04: SS 1 dB, downsample 8 kHz","E05: SS 3 dB, Opus 8 kbps"]):
        for detector in ["correlation","model_A","model_B"]:
            rows=[statistic(cid,detector,m) for m in frame_counts]
            ax.errorbar(frame_counts,[100*r["eer_mean"] for r in rows],yerr=[100*r["eer_std_sample"] for r in rows],fmt="o-",capsize=3,label=detector)
        ax.set(title=title,xlabel="Aggregated valid frames",ylabel="Test EER (%), model mean +/- seed SD",ylim=(0,100))
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle("Three model seeds; shared received WAVs, masks and groups; baseline measured once")
    fig.savefig(figures/"deep_test_eer.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout="constrained")
    for ax,row in zip(axes.flat,[r for r in training if r["seed"]==2026]):
        history=list(csv.DictReader((out/"training"/(row["name"]+"_history.csv")).open()))
        for field,label in [("train_bce","Train"),("validation_bce","Validation")]:ax.plot(range(1,26),[float(r[field]) for r in history],label=label)
        ax.axvline(row["best_epoch_diagnostic"],ls=":",color="tab:green",label="Best validation BCE (diagnostic)")
        ax.set(title=row["name"],xlabel="Epoch",ylabel="BCE loss");ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle("Seed 2026 training diagnostics; epoch 25 remains the primary result for all seeds")
    fig.savefig(figures/"training_seed2026.png",dpi=160);plt.close(fig)
    test_ss=q("SS",selection["ss_delta_db"]);test_lpc=q("SS_LPC",selection["lpc_delta_db"])
    lines=["# M5 替代语料主实验报告", "",
        f"运行 `{summary['experiment_id']}/{summary['run_id']}`。这是按预先固定子集开展的 E02～E05 方法复现，语料为 LibriSpeech，不能与论文 TIMIT 数值作同条件等价比较。", "",
        "## 数据和冻结顺序", "",
        "官方 train-clean-5、dev-clean-2 与 test-clean 下载包通过官方 MD5 和本地 SHA-256；原始选中 FLAC、许可证、SPEAKERS.TXT、来源/裁剪/准备音频哈希均保存。先按元数据及固定哈希选说话人与原始语句，再从每句居中截取单个 3 秒片段。选样不使用 VAD、PESQ 或检测结果。排除旧开发的 5703、3436、198 三位说话人。", "",
        "| 分区 | 语句/片段 | 说话人 | F/M 条数 | 秒 |", "| --- | ---: | ---: | --- | ---: |"]
    for split,a in summary["data_audit"]["audit"].items():lines.append(f"| {split} | {a['utterances']} | {a['speakers']} | {a['gender_utterances']['F']}/{a['gender_utterances']['M']} | {a['duration_s']:.0f} |")
    lines += ["", "训练来自 train-clean-5，验证来自 dev-clean-2，测试来自官方 test-clean，各自使用不同说话人。每条原语音只取一个片段，所有嵌入/通道/标签保持同分区；源文件、原始 PCM 和准备 PCM 的跨分区重复检查通过。F/M 仅引用官方元数据。子集裁剪与训练规模是本项目资源配置，不是官方完整测试榜单或完整 TIMIT 复现。", "",
        "验证集先选 SS/LPC 全局强度；再用训练集分别训练 A/B×C1/C3×3 种子，统一采用第 25 轮；验证集计算固定阈值。随后保存 evaluation_gate.json，才生成测试水印/通道、测试 PESQ 和检测结果。测试原文件的读取仅用于早期完整性与确定性裁剪，不参与选择模型或阈值。", "",
        "每源片段的 SS/LPC、0/1/2/3/4/6/8 dB 扫描共享一个增益，峰值上限 0.90，后续所有正负类和通道一致。共享噪声沿用 M2 的源 ID/种子生成规则。E04/E05 只用 SS，没有混入 LPC。", "",
        "## E02：通道前音质", "",
        "| 强度 dB | SS NB 均值 | LPC NB 均值 | SS WB 均值 | LPC WB 均值 |", "| ---: | ---: | ---: | ---: | ---: |"]
    for d in deltas:lines.append(f"| {d:g} | {q('SS',d).mean():.4f} | {q('SS_LPC',d).mean():.4f} | {q('SS',d,'pesq_wb').mean():.4f} | {q('SS_LPC',d,'pesq_wb').mean():.4f} |")
    lines += ["", "![通道前音质](figures/test_quality.png)", "", "误差条是这批语句的分布标准差，不是独立样本均值的置信区间。性别分组使用同一冻结清单：", "",
        "![按官方元数据分组的音质](figures/quality_by_gender.png)", "", "## E03：验证集匹配强度的测试鲁棒性", "",
        f"验证集选中 SS {selection['ss_delta_db']:g} dB / LPC {selection['lpc_delta_db']:g} dB，NB PESQ 均值 {selection['ss_mean_nb_pesq']:.6f}/{selection['lpc_mean_nb_pesq']:.6f}，差 {selection['absolute_mean_difference']:.6f}，容差 0.02。选择只使用验证 PESQ、不看 EER、不插值、不逐测试语音选强度。", "",
        f"同一强度应用到测试后，实测 NB PESQ 均值为 {test_ss.mean():.6f}/{test_lpc.mean():.6f}，差 {abs(test_ss.mean()-test_lpc.mean()):.6f}；不为了重新匹配测试音质而换强度。", "",
        "| 噪声 SNR | 有效帧 | SS EER % | LPC EER % | LPC−SS 百分点 |", "| --- | ---: | ---: | ---: | ---: |"]
    for noise in ["noise20","noise15"]:
        for count in [1,10,30]:
            a=statistic(f"E03_{noise}_SS","correlation",count);b=statistic(f"E03_{noise}_SS_LPC","correlation",count)
            lines.append(f"| {noise} | {count} | {100*a['eer_mean']:.3f} | {100*b['eer_mean']:.3f} | {100*(b['eer_mean']-a['eer_mean']):+.3f} |")
    lines += ["", "![LPC 噪声鲁棒性](figures/lpc_noise_test.png)", "", "## E04/E05：相关检测与 A/B", "",
        "A=6209 参数，B=15713 参数，B 卷积后无额外激活。两模型使用与相关检测相同的 63 维匹配对数谱位置，模型 float32，相关分数 float64；不额外标准化。20 ms 周期 Hann、10 ms 帧移、0.5 ms 倒谱截止、WebRTC VAD 2 均沿用固定方案。", "",
        "RMSprop lr=0.001、alpha=0.99、eps=1e-8、momentum/weight_decay=0、centered/foreach=false；BCEWithLogitsLoss，batch=1024、25 轮，CPU 单线程，三个种子 2026/2027/2028。验证最佳轮只作诊断，不替换第 25 轮。", "",
        "以下 A/B 为三个种子的均值±样本标准差（百分点）；传统相关检测无随机训练，只测一次。每条接收音频在三检测器及全部种子间共享有效帧与分组，线性 logit/相关分数分别求均值；有效帧可跨 VAD 间隙。", "",
        "| 条件 | 有效帧 | 相关 EER % | A EER % | B EER % | 负/正组数 | 配对语句/说话人 |", "| --- | ---: | ---: | ---: | ---: | --- | --- |"]
    for cid in ["C1","C3"]:
        for count in [1,10,30]:
            a=statistic(cid,"correlation",count);b=statistic(cid,"model_A",count);c=statistic(cid,"model_B",count)
            lines.append(f"| {cid} | {count} | {100*a['eer_mean']:.3f} | {100*b['eer_mean']:.3f} ± {100*b['eer_std_sample']:.3f} | {100*c['eer_mean']:.3f} ± {100*c['eer_std_sample']:.3f} | {a['negative_groups']}/{a['positive_groups']} | {a['paired_eligible_utterances']}/{a['speakers']} |")
    lines += ["", "完整 8 个聚合长度、逐种子值、每条语句首组 EER、参与语句/说话人和组数见 metrics.csv 与 seed_summary.csv。不把大量相关帧组当作同等数量的独立语句。", "",
        "![三种检测器测试 EER](figures/deep_test_eer.png)", "", "## 验证阈值的固定 FAR/FRR", "",
        "以下均为 30 个有效帧。阈值仅从 validation 估计，测试不重新校准；模型 FAR/FRR 为各自三个种子阈值下的均值，不意味着把阈值合成为一个通用阈值。", "",
        "| 条件 | 检测器 | 测试 FAR 均值 % | 测试 FRR 均值 % |", "| --- | --- | ---: | ---: |"]
    rates=list(csv.DictReader((out/"fixed_threshold_rates.csv").open()))
    for cid in ["C1","C3"]:
        for detector in ["correlation","model_A","model_B"]:
            v=[r for r in rates if r["condition"]==cid and r["split"]=="test" and r["detector"]==detector and int(r["frames"])==30]
            lines.append(f"| {cid} | {detector} | {100*np.mean([float(r['far']) for r in v]):.3f} | {100*np.mean([float(r['frr']) for r in v]):.3f} |")
    lines += ["", "## 差异解读", ""]
    for cid in ["C1","C3"]:
        base=statistic(cid,"correlation",30)
        for detector in ["model_A","model_B"]:
            differences=[statistic(cid,detector,m)["eer_mean"]-statistic(cid,"correlation",m)["eer_mean"] for m in frame_counts]
            r=statistic(cid,detector,30)
            lines.append(f"{cid}/{detector}：8 个长度中，三种子平均 EER 更低 {sum(x<0 for x in differences)} 项、相同 {sum(x==0 for x in differences)} 项、更高 {sum(x>0 for x in differences)} 项；30 帧相对相关检测变化 {100*(r['eer_mean']-base['eer_mean']):+.3f} 个百分点，种子标准差 {100*r['eer_std_sample']:.3f} 个百分点。设置间高度相关，不解释为独立胜率或显著性。")
    for noise in ["noise20","noise15"]:
        a=statistic(f"E03_{noise}_SS","correlation",1);b=statistic(f"E03_{noise}_SS_LPC","correlation",1)
        lines.append(f"{noise} 单帧，LPC−SS 为 {100*(b['eer_mean']-a['eer_mean']):+.3f} 个百分点；测试平均音质与验证配对的差异同时保留，不用检测结果重新选强度。")
    lines += ["", "这些差异只覆盖当前独立说话人的固定子集。训练来源较小、朗读材料与 3 秒裁剪、固定倒谱/VAD、LPC 实现细项和编解码版本均与论文存在条件差异；多帧的有效组数与源覆盖也会改变。训练/验证损失差距用于诊断，不据测试结果更换最佳轮。当前数据不支持将种子标准差解释为对所有新说话人的置信区间。", ""]
    lines += ["", "EER 是在每个评估分数总体上扫描阈值的描述性指标，与固定阈值 FAR/FRR 分开。校准覆盖的是此冻结子集与通道假设，独立任意上传仍返回 scores_only、decision=null，不启用该阈值。", "",
        "## 训练诊断与控制", "",
        "| 设置 | 末轮训练 BCE | 末轮验证 BCE | 最优验证轮（诊断） | 训练秒 |", "| --- | ---: | ---: | ---: | ---: |"]
    for r in training:lines.append(f"| {r['name']} | {r['last_train_bce']:.6f} | {r['last_validation_bce']:.6f} | {r['best_epoch_diagnostic']} | {r['fit_seconds']:.2f} |")
    control=json.loads((out/"shuffled_label_control.json").read_text())
    lines += ["", "![训练诊断](figures/training_seed2026.png)", "",
        f"另做 C1_A 文件标签打乱控制，训练文件标签改变 {control['changed_file_labels']} 项，验证保持真实标签。第 25 轮验证 EER 单帧 {100*control['validation']['eer_1']:.3f}%、30 帧 {100*control['validation']['eer_30']:.3f}%；这是一次负对照诊断，不根据它改主模型，也不假定有限数据必然恰为 50%。", "",
        f"主实验至审计前用时 {summary['elapsed_s_before_audit']:.2f} 秒；进程累计峰值 RSS {summary['peak_process_rss_bytes']/1024**2:.1f} MiB，非独立模型显存。具体训练速度只适用于本机和本次子集，不能推算全 TIMIT。", "",
        "## 核验和边界", "",
        f"- {validation['dataset_validation']['utterances_verified']} 个原始/裁剪语句完整性、说话人/源/PCM 隔离核验。",
        f"- 重算 {validation['embedded_wavs_recomputed']} 个嵌入 WAV，核验 {validation['received_and_feature_caches_verified']} 个接收 WAV/缓存、{validation['opus_verified']} 个实际 Opus、{validation['noise_wavs_recomputed']} 个共享噪声 WAV。",
        f"- 12 个主 checkpoint 严格加载/哈希核验；种子 2026 的 {len(validation['fully_retrained_settings'])} 个设置完整重训，张量 SHA/逐轮历史一致，其余两种子未完整重训。所有 12 个模型均重新推理、重算保存的帧/组分数。",
        f"- {validation['metric_rows_recomputed']} 条 EER 由已审计 CSV 重算；{validation['validation_thresholds_verified']} 个阈值均来自 validation；{validation['zero_eer_settings']} 个零强度 EER 为 50%。",
        f"- PESQ 在分区×性别×方法×强度的 {validation['pesq_strata_recomputed']} 个分层样例重算，最大差 {validation['pesq_maximum_error']:g}；其余音质行的输入嵌入 WAV 全量重算，不声称全部 PESQ 重跑。",
        f"- 最低有效帧 {validation['minimum_valid_frames']}；需共同衰减的源片段 {validation['attenuated_source_clips']}。覆盖和 PESQ 失败保留，未以假分数填补。M1～M4 冻结材料哈希保持原值。", "",
        "本轮采用独立测试说话人的固定替代语料子集，补齐了早期 smoke 缺少的训练/验证/测试闭环。训练规模、朗读来源、单句裁剪、元数据和 TIMIT 均不同；作者的百分比降幅仅作论文参考，不能写成本项目已同条件复现的成绩。正负结果和种子差异完整保留，结果不用于回调测试集参数。后续 M6 探索、M7 界面/云端和最终报告另行推进。", "",
        "## 复查材料", "",
        "input_manifest.jsonl、dataset_freeze.json、dataset_validation.json、config.yaml、environment.json、codec_runtime.json、key.json；quality_scores.jsonl/csv、quality_selection.json、quality_gate.json；training/、training_summary.json、shuffled_label_control.json、checkpoints/、evaluation_gate.json、calibration.json；各 audio/feature/channel/reference manifest、coverage.jsonl；validation/test_frame_scores.csv、aggregate_scores.csv、group_definitions.csv、threshold_curves.csv；metrics.json/csv、seed_summary.csv、fixed_threshold_rates.csv、validation.json、failures.jsonl、run_status.json。", "",
        "语料来源：[OpenSLR LibriSpeech SLR12](https://www.openslr.org/12/)、[官方 Mini LibriSpeech SLR31](https://www.openslr.org/31/)，许可 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。源包完整 MD5 与 SHA-256 在 data/manifests/librispeech_m5_v1/sources.json；所用全部原始 FLAC 与许可证进入归档，完整未选语句的下载包仅留本地缓存。", ""]
    path=out/"M5主实验报告.md";path.write_text("\n".join(lines));return path
