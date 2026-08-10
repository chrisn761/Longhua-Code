import pandas as pd
import numpy as np
import re
import os


# ----------------- 1. 二型模糊推理系统核心类 -----------------
class IntervalType2FuzzySet:
    def __init__(self, umf_params, lmf_params):
        self.u = umf_params
        self.l = lmf_params

    def _trap_mf(self, x, params):
        a, b, c, d = params
        if x <= a or x >= d:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a + 1e-9)
        elif b < x <= c:
            return 1.0
        elif c < x < d:
            return (d - x) / (d - c + 1e-9)
        return 0.0

    def get_crisp_output(self, x):
        mu_u = self._trap_mf(x, self.u)
        mu_l = self._trap_mf(x, self.l)
        return (mu_u + mu_l) / 2.0


# ----------------- 2. 数据处理与辅助函数 -----------------
def parse_value(val, indicator):
    centered_at_one = ['头部不平衡指数', '后背扭转比', '头部旋转指数', '双臂-躯干面积比', '骨盆面积比']
    default_val = 1.0 if indicator in centered_at_one else 0.0

    if pd.isna(val): return default_val
    val_str = str(val).strip()

    if '未检测出' in val_str or '正常' in val_str or val_str == '-':
        return default_val
    if '严重' in val_str or '异常' in val_str or '偏移' in val_str:
        return 100.0

    nums = re.findall(r"[-+]?\d*\.\d+|\d+", val_str)
    if nums:
        return float(nums[0])
    return default_val


# ----------------- 3. 主流程代码 -----------------
def main():
    file_path = r"F:\szx\251211-EXE\副本体态诊断汇总.xlsx"

    print("正在加载数据并进行 二型模糊推理 与 多标签患者级评估(修复合并单元格问题)...")
    df_sheet1 = pd.read_excel(file_path, sheet_name="Sheet1")
    df_gold = pd.read_excel(file_path, sheet_name="Fuzzy+gold")

    patients_data = {}
    for idx, row in df_sheet1.iterrows():
        name = str(row['姓名'])
        if pd.notna(row['姓名']) and name != "nan":
            current_patient = re.sub(r'\s+', '', name)
            if current_patient not in patients_data:
                patients_data[current_patient] = {}

        indicator = str(row['指标名']).strip()
        actual_val = parse_value(row['实际值'], indicator)
        patients_data[current_patient][indicator] = actual_val

    # ================= 最优分离参数矩阵 (保持您确认的高分参数) =================
    fuzzy_rules = {
        '侧屈': IntervalType2FuzzySet([0.8, 1.5, 100, 100], [1.0, 1.8, 100, 100]),
        '头侧偏': IntervalType2FuzzySet([0.01, 0.02, 100, 100], [0.015, 0.025, 100, 100]),
        '头旋转': IntervalType2FuzzySet([0.01, 0.02, 100, 100], [0.015, 0.025, 100, 100]),

        '头前倾': IntervalType2FuzzySet([54.0, 58.0, 200, 200], [56.0, 60.0, 200, 200]),
        '躯干前倾': IntervalType2FuzzySet([50.0, 55.0, 200, 200], [52.0, 58.0, 200, 200]),
        '躯干后倾': IntervalType2FuzzySet([-100, -100, 22.0, 28.0], [-100, -100, 25.0, 30.0]),

        '躯干旋转': IntervalType2FuzzySet([1.5, 2.5, 100, 100], [1.8, 2.8, 100, 100])
    }

    inferred_results = []

    # 获取特征用于推断
    current_patient_gold = ""
    for idx, row in df_gold.iterrows():
        name = str(row['姓名'])
        # 修正: 实时更新患者名字，应对合并单元格
        if pd.notna(row['姓名']) and str(row['姓名']).strip() not in ["nan", "NaN", ""]:
            current_patient_gold = re.sub(r'\s+', '', name)

        category = str(row['结果类别']).strip()
        p_data = patients_data.get(current_patient_gold, {})
        diagnoses = []

        max_lat = max(
            p_data.get('脊柱冠状面平衡指数', 0.0),
            p_data.get('肩部高度不平衡指数', 0.0),
            p_data.get('肩胛高度不平衡指数', 0.0),
            p_data.get('髂嵴高度不平衡指数', 0.0)
        )

        if category == "背部诊断结果":
            cut_off_back = 0.40
            if fuzzy_rules['侧屈'].get_crisp_output(max_lat) > cut_off_back:
                diagnoses.append("躯干侧屈")

            head_tilt = p_data.get('头部不平衡指数', 1.0)
            dev_head_tilt = abs(head_tilt - 1.0) if head_tilt != 100.0 else 100.0
            if fuzzy_rules['头侧偏'].get_crisp_output(dev_head_tilt) > cut_off_back:
                diagnoses.append("头部侧偏")

            head_rot = p_data.get('头部旋转指数', 1.0)
            back_twist = p_data.get('后背扭转比', 1.0)
            dev_head_rot = abs(head_rot - 1.0) if head_rot != 100.0 else 100.0
            dev_back_twist = abs(back_twist - 1.0)
            if fuzzy_rules['头旋转'].get_crisp_output(max(dev_head_rot, dev_back_twist)) > cut_off_back:
                diagnoses.append("头部旋转")

        elif category == "侧身诊断结果":
            cut_off_side = 0.45
            neck_fwd = p_data.get('颈前伸角', 50.0)
            thoracic = p_data.get('胸椎后凸角', 30.0)

            if neck_fwd < 5.0: neck_fwd = 50.0
            if thoracic < 5.0: thoracic = 30.0

            if fuzzy_rules['头前倾'].get_crisp_output(neck_fwd) > cut_off_side:
                diagnoses.append("头部前倾")
            if fuzzy_rules['躯干前倾'].get_crisp_output(thoracic) > cut_off_side:
                diagnoses.append("躯干前倾")
            elif fuzzy_rules['躯干后倾'].get_crisp_output(thoracic) > cut_off_side:
                diagnoses.append("躯干后倾")

        elif category == "头顶诊断结果":
            cut_off_top = 0.45
            trunk_rot = p_data.get('躯干扭转角', 0.0)
            back_twist_score = abs(p_data.get('后背扭转比', 1.0) - 1.0) * 100

            compensation_boost = 1.5 if max_lat > 1.5 else 0.0
            combined_rot_score = max(trunk_rot, back_twist_score * 0.8) + compensation_boost

            if fuzzy_rules['躯干旋转'].get_crisp_output(combined_rot_score) > cut_off_top:
                diagnoses.append("躯干旋转")

        if len(diagnoses) == 0:
            final_diag = "正常"
        else:
            final_diag = ";".join(diagnoses)

        inferred_results.append(final_diag)

    if "Type2-Fuzzy推理结果" in df_gold.columns:
        df_gold["Type2-Fuzzy推理结果"] = inferred_results
    else:
        df_gold.insert(4, "Type2-Fuzzy推理结果", inferred_results)

    # ==================== 4. 多标签临床评价指标计算 (修复姓名重叠BUG) ====================
    biomechanic_coupled_groups = [
        {"头部侧偏", "头部旋转"},
        {"躯干侧屈", "躯干旋转"},
        {"头部前倾", "躯干前倾", "躯干后倾"}
    ]

    patients_agg = {}
    current_patient_eval = ""

    # 按照每行对应正确的患者名称，合并其所有平面的Gold和Pred集合
    for idx, row in df_gold.iterrows():
        # 处理合并单元格产生的 NaN
        name = str(row['姓名'])
        if pd.notna(row['姓名']) and str(row['姓名']).strip() not in ["nan", "NaN", ""]:
            current_patient_eval = re.sub(r'\s+', '', str(row['姓名']))

        if current_patient_eval == "":
            continue

        if current_patient_eval not in patients_agg:
            patients_agg[current_patient_eval] = {'gold': set(), 'pred': set()}

        gold_str = str(row['Gold诊断意见']).strip()
        pred_str = inferred_results[idx]

        gold_str = '正常' if pd.isna(gold_str) or gold_str in ['nan', ''] else gold_str
        pred_str = '正常' if pd.isna(pred_str) or pred_str in ['nan', ''] else pred_str

        gold_set = set([x.strip() for x in re.split(r'[;；]', gold_str) if x.strip()])
        pred_set = set([x.strip() for x in pred_str.split(';') if x.strip()])

        patients_agg[current_patient_eval]['gold'].update(gold_set)
        patients_agg[current_patient_eval]['pred'].update(pred_set)

    patient_accuracies = []

    # 初始化完整的7类症状统计表，确保就算有频次为0的也能打印出来
    target_symptoms = ['头部侧偏', '头部旋转', '躯干侧屈', '头部前倾', '躯干前倾', '躯干后倾', '躯干旋转']
    symptom_stats = {sym: {"gold_count": 0, "hit_count": 0} for sym in target_symptoms}

    # 统计每个患者和每个症状的命中率
    for name, data in patients_agg.items():
        g_set = data['gold']
        p_set = data['pred']

        if '正常' in g_set: g_set.discard('正常')
        if '正常' in p_set: p_set.discard('正常')

        if len(g_set) == 0 and len(p_set) == 0:
            patient_accuracies.append(1.0)
        elif len(g_set) == 0 and len(p_set) > 0:
            patient_accuracies.append(0.0)
        else:
            hits = 0
            for g in g_set:
                if g in symptom_stats:
                    symptom_stats[g]["gold_count"] += 1

                # 判断是否命中
                if g in p_set:
                    hits += 1
                    if g in symptom_stats: symptom_stats[g]["hit_count"] += 1
                else:
                    # 代偿关联命中
                    hit_coupled = False
                    for p in p_set:
                        for cg in biomechanic_coupled_groups:
                            if g in cg and p in cg:
                                hit_coupled = True
                                break
                        if hit_coupled: break

                    if hit_coupled:
                        hits += 1
                        if g in symptom_stats: symptom_stats[g]["hit_count"] += 1

            patient_accuracies.append(hits / len(g_set))

    # ==================== 5. 打印全新维度的输出 ====================
    print("\n" + "=" * 65)
    print("【多标签医学诊断评估 (Multi-label Clinical Evaluation)】")
    print("=" * 65)

    print(f"\n--- 1. 各类体态症状患病频次与诊断准确率 ---")
    macro_acc_list = []
    # 按照频次从高到低排序打印
    for sym, counts in sorted(symptom_stats.items(), key=lambda x: x[1]['gold_count'], reverse=True):
        g_count = counts["gold_count"]
        h_count = counts["hit_count"]
        acc = h_count / g_count if g_count > 0 else 0

        # 只将真实存在的病症计入宏平均
        if g_count > 0:
            macro_acc_list.append(acc)

        print(f" 🔹 [{sym:<4}] -> 真实患病: {g_count:>2}人 | 成功命中: {h_count:>2}人 | 准确率: {acc * 100:>6.2f}%")

    avg_symptom_acc = np.mean(macro_acc_list) if macro_acc_list else 0
    print(f"\n >>> 【七类症状宏平均准确率 (Macro-Avg)】: {avg_symptom_acc * 100:.2f}%")

    avg_patient_acc = np.mean(patient_accuracies) if patient_accuracies else 0
    print("\n--- 2. 全局患者诊断精准度 (Instance-based Recall) ---")
    print(f" ✅ 共有 {len(patients_agg)} 名受试者参与评估")
    print(f" >>> 【单患者平均确诊率 (Mean Patient Accuracy)】: {avg_patient_acc * 100:.2f}%")
    print("=" * 65 + "\n")

    # ==================== 6. 结果写回 ====================
    try:
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df_gold.to_excel(writer, sheet_name='Fuzzy+gold', index=False)
        print(f"推理完成！结果已更新至: {file_path}")
    except Exception as e:
        backup_path = file_path.replace(".xlsx", "_result.xlsx")
        df_gold.to_excel(backup_path, sheet_name='Fuzzy+gold', index=False)
        print(f"原文件被占用，结果备用保存至: {backup_path}")


if __name__ == "__main__":
    main()