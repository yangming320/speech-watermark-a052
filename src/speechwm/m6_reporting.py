from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .utils import read_jsonl


def make_extension_report(out:Path):
    summary=json.loads((out/'metrics.json').read_text());proof=json.loads((out/'validation.json').read_text())
    rows=list(csv.DictReader((out/'seed_summary.csv').open()));rates=list(csv.DictReader((out/'fixed_threshold_rates.csv').open()));figures=out/'figures';figures.mkdir(exist_ok=True)
    def stat(cid,d,n=30):return next(r for r in rows if r['condition']==cid and r['detector']==d and int(r['frames'])==n)
    def value(cid,d,n=30):return 100*float(stat(cid,d,n)['eer_mean'])
    detectors=['correlation','single_A','single_B','mixed_A','mixed_B'];bitrates=[6,8,12,16]
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,n in zip(axes,[1,10,30]):
        for d in detectors:
            data=[stat(f'R{r}',d,n) for r in bitrates]
            ax.errorbar(bitrates,[100*float(r['eer_mean']) for r in data],yerr=[100*float(r['eer_std_sample']) for r in data],fmt='o-',capsize=2,label=d)
        ax.set(title=f'{n} valid frames',xlabel='Actual Opus payload rate (kbps)',ylabel='Test EER (%)',ylim=(0,60));ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('E07 / E08: fixed 8 kbps versus equal-budget mixed-rate training, three seeds')
    fig.savefig(figures/'cross_rate_eer.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,field in zip(axes,['far','frr']):
        for d in detectors:
            vals=[100*np.mean([float(r[field]) for r in rates if r['condition']==f'R{rate}' and r['split']=='test' and r['detector']==d and int(r['frames'])==30]) for rate in bitrates]
            ax.plot(bitrates,vals,'o-',label=d)
        ax.set(title=f'Fixed validation threshold: {field.upper()}',xlabel='Opus rate (kbps)',ylabel='Test rate (%)',ylim=(0,100));ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.savefig(figures/'cross_rate_fixed_errors.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,channel in zip(axes,['D','O']):
        for method in ['SS','SS_LPC']:
            for d in ['correlation','combo_A','combo_B']:
                ax.plot([1,10,30],[value(f'E06_{channel}_{method}',d,n) for n in [1,10,30]],'o-',label=f'{method} / {d}')
        ax.set(title='Downsample 8 kHz' if channel=='D' else 'Opus 8 kbps',xlabel='Valid frames per group',ylabel='Test EER (%)',ylim=(0,60));ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.suptitle('E06: validation-selected SS 2 dB / LPC 6 dB, compatible models and matched training budgets')
    fig.savefig(figures/'lpc_deep_combination.png',dpi=150);plt.close(fig)
    offsets=[0,2.5,5,10,15,20];fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    for d in detectors:ax.plot(offsets,[value(f'OFFSET_{x:g}',d) for x in offsets],'o-',label=d)
    ax.set(title='E09: known starting crop, same received Opus 8 kbps WAV',xlabel='Samples removed at start (ms)',ylabel='30-frame test EER (%)',ylim=(0,60));ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(figures/'start_offset_eer.png',dpi=150);plt.close(fig)
    coverage=read_jsonl(out/'coverage.jsonl');fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for label in [0,1]:
        vals=[];spectrum=[]
        for rate in bitrates:
            selected=[r for r in coverage if r['condition']==f'R{rate}' and r['split']=='test' and r['kind']=='main' and r['label']==label]
            vals.append(np.mean([r['valid_frames'] if 'valid_frames' in r else r['voiced_frames'] for r in selected]))
            spectrum.append(np.mean([r['mean_abs_matched_feature'] for r in selected]))
        axes[0].plot(bitrates,vals,'o-',label=f'label {label}');axes[1].plot(bitrates,spectrum,'o-',label=f'label {label}')
    axes[0].set(ylabel='Mean valid frames / received file');axes[1].set(ylabel='Mean absolute matched feature')
    for ax in axes:ax.set(xlabel='Opus rate (kbps)');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Same test utterances: label-specific VAD coverage and feature scale')
    fig.savefig(figures/'coverage_feature_scale.png',dpi=150);plt.close(fig)
    lines=['# M6 E06～E09 扩展实验报告','','本轮沿用 M5 的 LibriSpeech 固定替代子集，288/80/200 句，24/10/40 位说话人隔离；同一 M5 测试集上的探索，不称为新的未触及确认集。协议在扩展测试评分前固定，训练不取 test，第 25 轮不按测试挑选。','','新增 30 个 A/B 模型，三种子 2026/2027/2028，另逐字节复用 M5/C3 的 6 个固定 8 kb/s 主模型；相关检测只测一次。模型均值±样本标准差只衡量训练种子波动，不能解释为新说话人总体置信区间。','','## E07/E08：跨码率与混合训练','','SS 3 dB、8 kHz、实际 Opus 硬 CBR 6/8/12/16 kb/s，20 ms 包字节数分别为 15/20/30/40。其余编解码选项保持一致。M5 的 8 kb/s 接收 WAV 和六权重逐字节拷贝。','','混合训练按标签在四码率等额无放回抽帧，总帧数和每轮更新次数与原 8 kb/s 训练严格相同；架构、初始化种子、优化器、batch=1024 和 25 轮不变。抽样来源、有效帧索引和码率比例见 training_selection.jsonl / training_budget.json。训练诊断 validation 仍为 8 kb/s，不挑最佳轮；最终混合阈值来自四码率 validation 池。','','| 测试 kb/s，30 帧 | 相关 EER % | 单码率 A % | 单码率 B % | 混合 A % | 混合 B % |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for rate in bitrates:lines.append('| '+str(rate)+' | '+' | '.join(f'{value(f"R{rate}",d):.3f}' for d in detectors)+' |')
    lines+=['','![跨码率 EER](figures/cross_rate_eer.png)','','单码率模型及相关检测的固定阈值始终继承 M5/8 kb/s validation；混合模型每种子/聚合长度使用一个四码率 pooled validation 阈值。各码率 validation EER 阈值另作诊断保存在 calibration.json，不替换主固定阈值。两模型族阈值校准的通道覆盖不同，因此固定 FAR/FRR 的差异包含这一设计。','','![固定阈值误差](figures/cross_rate_fixed_errors.png)','','| 检测器 | 6 kb/s 相对 8 kb/s EER 变化，百分点 |','| --- | ---: |']
    for d in detectors:lines.append(f'| {d} | {value("R6",d)-value("R8",d):+.3f} |')
    lines+=['','| 架构 | 混合−单码率 EER 百分点：6 / 8 / 12 / 16 kb/s，30 帧 |','| --- | --- |']
    for arch in ['A','B']:lines.append(f'| {arch} | '+' / '.join(f'{value(f"R{r}","mixed_"+arch)-value(f"R{r}","single_"+arch):+.3f}' for r in bitrates)+' |')
    lines+=['','负数表示本次混合模型 EER 更低，正数表示代价；全部四个码率保留，不按最优码率叙述整体成功。测试 EER 为描述性阈值扫描，固定阈值 FAR/FRR 另表列出。','','![覆盖与特征尺度](figures/coverage_feature_scale.png)','','正负类覆盖和特征尺度同时保留，用于区分 VAD 组成变化、特征尺度及分类能力，当前观察不单独证明因果机制。','','## E06：验证强度配对下的 LPC 与兼容深度检测','','固定 M5 validation 选定的 SS 2 dB / LPC 6 dB；同一训练语句/标签在四个方法×通道条件中取共同最小有效帧数，无放回抽取，故逐文件帧预算、总帧和优化步数相同。每个条件分别训练兼容 A/B 三种子，共 24 个新设置。validation/test 使用全部有效帧，无测试挑强度。','','| 通道 | 方法 | 相关 EER % | 兼容 A % | 兼容 B % |','| --- | --- | ---: | ---: | ---: |']
    for c in ['D','O']:
        for method in ['SS','SS_LPC']:lines.append(f'| {c} | {method} | '+' | '.join(f'{value(f"E06_{c}_{method}",d):.3f}' for d in ['correlation','combo_A','combo_B'])+' |')
    lines+=['','D=降采样 8 kHz，O=8 kHz→Opus 8 kb/s。上述表均为 30 帧；1/10 帧、逐种子标准差、首组 EER、参与组数/语句/说话人和固定 FAR/FRR 见 CSV。SS2/LPC6 的通道前音质引用 m5_quality_scores.jsonl，接收音质 quality_scores.jsonl 包含通道总失真；测试 SS2/LPC6 的 NB PESQ 为 4.467317/4.419386，差 0.047931，超过验证容差 0.02。因此本表是验证配对强度的测试迁移，不能称严格等测试音质实验；不据测试重新匹配。','','![LPC 深度组合](figures/lpc_deep_combination.png)','','## E09：已知起点偏移','','对相同 R8 接收 WAV 裁去起点 0/2.5/5/10/15/20 ms，重算 VAD 与特征；不补零、不重编码、不搜索最高分。固定阈值继承未偏移条件，观测长度略变，不能把全部变化只归因于相位。0 ms 与 R8 指标/逐组分数核对一致。','','| 起点裁去 ms，30 帧 | 相关 EER % | 单 A % | 单 B % | 混合 A % | 混合 B % |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for o in offsets:lines.append('| '+str(o)+' | '+' | '.join(f'{value(f"OFFSET_{o:g}",d):.3f}' for d in detectors)+' |')
    lines+=['','![起点偏移](figures/start_offset_eer.png)','','这是受控已知偏移敏感性分析，没有实现未知位置同步恢复。多偏移取最大值会改变误报，不能沿用本轮固定阈值。独立上传仍返回 scores_only、decision=null。','','## 验证与复查','','']
    for key in ['embedded_wavs_recomputed','received_and_features_recomputed','opus_streams_verified','m5_r8_received_reused_exactly','offset_crops_verified','zero_eer_settings','checkpoints_verified','metric_rows_recomputed','fixed_threshold_rate_rows','received_pesq_strata_recomputed','minimum_valid_frames']:lines.append(f'- {key}: {proof[key]}')
    if proof.get('resolved_initial_audit_failure'):
        lines+=['',f"首次审计的接收峰值 <1 断言不适用于浮点 Opus 解码：{len(proof['floating_opus_overshoots_redecoded'])} 条 R6 音频轻微超出全幅，最大 {proof['maximum_received_float_peak']:.9f}。已对这四条原码流重新解码并逐点确认，保留浮点音频、全部权重、阈值、特征和分数，没有削波或重新缩放。failures.jsonl 保留原审计失败，audit_recovery.json 绑定未变主文件与修正；后续完整审计通过。"]
    lines+=['',f"种子 2026 十个新设置完整重训：{', '.join(proof['fully_retrained_settings'])}。其余新种子未完整重训；全部 36 权重严格加载并重新推理，全部帧预测缓存和聚合 CSV 对齐重算。PESQ 仅分层样例重算，其他音质行的输入音频完整核验。",'',f"主运行审计前用时 {summary['elapsed_s_before_audit']:.1f} 秒。本机 CPU 单线程；不推断云端速度。M1～M5 快照保持固定。",'','完整记录：config.yaml、input_manifest.jsonl、m5_dependency.json、training_budget.json、training_selection.jsonl、training/、training_summary.json、checkpoints/、evaluation_gate.json、calibration.json、各 manifest、coverage.jsonl、quality_scores.jsonl、frame_predictions/、*_aggregate_scores.csv、metrics.csv、seed_summary.csv、fixed_threshold_rates.csv、validation.json、failures.jsonl。模型和图表均对应同一运行。','']
    path=out/'M6扩展实验报告.md';path.write_text('\n'.join(lines));return path
