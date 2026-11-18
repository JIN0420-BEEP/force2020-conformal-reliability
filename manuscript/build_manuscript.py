# -*- coding: utf-8 -*-
"""生成《物探化探计算技术》投稿用中文论文 Word 初稿。"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
OUT = ROOT / "manuscript" / "FORCE2020_岩性识别可靠性_初稿.docx"

# 字体
HEI = "黑体"
SONG = "宋体"
KAI = "楷体"
TNR = "Times New Roman"


# ---------------------------------------------------------------- 基础工具
def set_run(run, cn=SONG, en=TNR, size=10.5, bold=False, color=None):
    run.font.name = en
    run.font.size = Pt(size)
    run.font.bold = bold
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), cn)
    rfonts.set(qn("w:ascii"), en)
    rfonts.set(qn("w:hAnsi"), en)
    if color is not None:
        run.font.color.rgb = RGBColor(*color)


def para(doc, text="", cn=SONG, en=TNR, size=10.5, bold=False,
         align=None, indent_chars=2, space_before=0, space_after=0,
         line=1.5, color=None):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    if align is not None:
        p.alignment = align
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = line
    if indent_chars:
        ind = p._p.get_or_add_pPr().find(qn("w:ind"))
        if ind is None:
            ind = OxmlElement("w:ind")
            p._p.get_or_add_pPr().append(ind)
        ind.set(qn("w:firstLineChars"), str(indent_chars * 100))
        ind.set(qn("w:firstLine"), str(int(size * 2 * indent_chars)))
    if text:
        r = p.add_run(text)
        set_run(r, cn=cn, en=en, size=size, bold=bold, color=color)
    return p


def heading(doc, text, level=1):
    sizes = {1: 12, 2: 10.5, 3: 10.5}
    p = para(doc, text, cn=HEI, en=TNR, size=sizes.get(level, 10.5),
             bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, indent_chars=0,
             space_before=10 if level == 1 else 6, space_after=4, line=1.5)
    return p


def caption(doc, text):
    return para(doc, text, cn=SONG, en=TNR, size=9, bold=False,
                align=WD_ALIGN_PARAGRAPH.CENTER, indent_chars=0,
                space_before=3, space_after=6, line=1.0)


def add_figure(doc, filename, cap, width_cm=13.5):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.0
    run = p.add_run()
    run.add_picture(str(FIG / filename), width=Cm(width_cm))
    caption(doc, cap)


def _set_cell_border(cell, **kwargs):
    tc = cell._tc
    tcpr = tc.get_or_add_tcPr()
    borders = tcpr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcpr.append(borders)
    for edge in ("top", "bottom", "left", "right"):
        spec = kwargs.get(edge)
        tag = borders.find(qn(f"w:{edge}"))
        if tag is None:
            tag = OxmlElement(f"w:{edge}")
            borders.append(tag)
        if spec is None:
            tag.set(qn("w:val"), "nil")
        else:
            tag.set(qn("w:val"), "single")
            tag.set(qn("w:sz"), str(spec))
            tag.set(qn("w:space"), "0")
            tag.set(qn("w:color"), "000000")


def three_line_table(doc, headers, rows, col_align=None):
    n_cols = len(headers)
    table = doc.add_table(rows=1, cols=n_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    # 表头
    hdr = table.rows[0].cells
    for j, h in enumerate(headers):
        hdr[j].text = ""
        p = hdr[j].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.line_spacing = 1.0
        r = p.add_run(h)
        set_run(r, size=9, bold=True)
    # 数据
    for row in rows:
        cells = table.add_row().cells
        for j, val in enumerate(row):
            cells[j].text = ""
            p = cells[j].paragraphs[0]
            p.paragraph_format.line_spacing = 1.0
            if col_align and col_align[j] == "l":
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            elif col_align and col_align[j] == "r":
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(str(val))
            set_run(r, size=9, bold=False)
    # 三线表边框
    n_rows = len(table.rows)
    for j in range(n_cols):
        for i in range(n_rows):
            kw = {"left": None, "right": None}
            if i == 0:
                kw["top"] = 12
                kw["bottom"] = 6
            elif i == n_rows - 1:
                kw["bottom"] = 12
            _set_cell_border(table.rows[i].cells[j], **kw)
    return table


def table_caption(doc, text):
    return para(doc, text, cn=SONG, en=TNR, size=9, bold=False,
                align=WD_ALIGN_PARAGRAPH.CENTER, indent_chars=0,
                space_before=8, space_after=2, line=1.0)


# ================================================================ 构建文档
doc = Document()

# 页面：A4，四边 2.5 cm
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.top_margin = sec.bottom_margin = Cm(2.5)
sec.left_margin = sec.right_margin = Cm(2.5)

# Normal 默认样式
normal = doc.styles["Normal"]
normal.font.name = TNR
normal.font.size = Pt(10.5)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), SONG)

# ------------------------------------------------ 中文题名
para(doc, "空间分块与选择性预测下测井岩性识别可靠性评价",
     cn=HEI, en=TNR, size=18, bold=True,
     align=WD_ALIGN_PARAGRAPH.CENTER, indent_chars=0,
     space_before=0, space_after=4, line=1.3)
para(doc, "——以 FORCE 2020 数据集为例",
     cn=HEI, en=TNR, size=14, bold=True,
     align=WD_ALIGN_PARAGRAPH.CENTER, indent_chars=0,
     space_after=8, line=1.3)

# 作者与单位（双盲评审占位，投稿前替换）
para(doc, "[第一作者姓名]，[第二作者姓名]，[通信作者姓名]",
     cn=KAI, en=TNR, size=12, align=WD_ALIGN_PARAGRAPH.CENTER,
     indent_chars=0, space_after=2, line=1.2)
para(doc, "([作者单位全称，省 市 邮编])",
     cn=SONG, en=TNR, size=9, align=WD_ALIGN_PARAGRAPH.CENTER,
     indent_chars=0, space_after=8, line=1.2)

# ------------------------------------------------ 中文摘要
abs_cn = (
    "针对测井岩性机器学习识别普遍采用随机深度点交叉验证、其结果可能因井内与空间自相关而"
    "高估真实泛化能力的问题，系统评估不同验证协议下公开测井岩性识别的可靠性，以及置信度"
    "拒判机制向新井、新区域迁移的有效性。以 FORCE 2020 公开数据集（北海 118 口井、10 类"
    "岩性、143 万余个有标签深度点）为对象，构建随机深度点、按井分组和带 20 km 缓冲的空间"
    "分块三套五折交叉验证协议，采用直方图梯度提升、随机森林和 LightGBM 三类模型，并引入"
    "嵌套校准的选择性预测及连续井段的岩性边界、层厚评价。结果表明：直方图梯度提升模型的"
    "宏平均 F1 由随机划分的 0.788 降至按井分组的 0.424 和空间分块的 0.301，白垩、泥灰岩"
    "等少数类召回率在空间外推下坍缩至接近零；目标精度 90% 的拒判阈值在随机划分下五折全部"
    "达标，而在跨井与空间外推的 15 折仅 3 折达标；连续井段的岩性边界 F1 仅约 0.29，层厚"
    "相对误差达 41.5%。研究表明，随机深度点验证会显著高估模型的现场部署性能，训练域校准的"
    "置信度阈值难以直接迁移到新井与新区域；岩性识别应同时报告空间外推指标、少数类表现与"
    "连续边界精度，并将低置信度井段交由人工复核。"
)
p = para(doc, "", indent_chars=0, space_after=2, line=1.3)
r = p.add_run("摘要：")
set_run(r, cn=HEI, size=9, bold=True)
r = p.add_run(abs_cn)
set_run(r, cn=SONG, size=9)

p = para(doc, "", indent_chars=0, space_after=2, line=1.3)
r = p.add_run("关键词：")
set_run(r, cn=HEI, size=9, bold=True)
r = p.add_run("测井；岩性识别；空间交叉验证；选择性预测；置信度校准；FORCE 2020")
set_run(r, cn=SONG, size=9)

p = para(doc, "", indent_chars=0, space_after=6, line=1.3)
r = p.add_run("中图分类号：")
set_run(r, cn=HEI, size=9, bold=True)
r = p.add_run("P631.84    文献标志码：A")
set_run(r, cn=SONG, size=9)

# ------------------------------------------------ 英文前置
para(doc, "Reliability evaluation of well-log lithofacies classification under "
          "spatial blocking and selective prediction: a case study of the "
          "FORCE 2020 dataset",
     cn=HEI, en=TNR, size=13, bold=True,
     align=WD_ALIGN_PARAGRAPH.CENTER, indent_chars=0,
     space_before=8, space_after=6, line=1.3)
para(doc, "[First Author], [Second Author], [Corresponding Author]",
     cn=SONG, en=TNR, size=10.5, align=WD_ALIGN_PARAGRAPH.CENTER,
     indent_chars=0, space_after=2, line=1.2)
para(doc, "([Affiliation, City, Postcode, China])",
     cn=SONG, en=TNR, size=9, align=WD_ALIGN_PARAGRAPH.CENTER,
     indent_chars=0, space_after=6, line=1.2)

abs_en = (
    "Machine-learning lithofacies classification from well logs is commonly validated "
    "with random depth-point cross-validation, whose reported performance can be strongly "
    "optimistic because within-well and spatial dependence are ignored. This study evaluates "
    "the reliability of open well-log lithofacies classification under different validation "
    "protocols, and examines whether a confidence-based rejection rule calibrated in one "
    "domain can be transferred to unseen wells and unseen geographic regions. Using the "
    "public FORCE 2020 dataset, which comprises 118 wells, ten lithofacies classes and more "
    "than 1.43 million labelled depth points in the North Sea, three five-fold cross-validation "
    "protocols were designed: random depth-point splits, well-grouped splits and spatial-block "
    "splits with a 20 km exclusion buffer. Three classifiers, namely histogram gradient "
    "boosting, random forest and LightGBM, were compared, together with nested calibration of "
    "selective prediction and continuous-depth evaluation of lithological boundaries and bed "
    "thickness. For histogram gradient boosting, the macro-averaged F1 score dropped from 0.788 "
    "under random splitting to 0.424 under well-grouped splitting and 0.301 under spatial-block "
    "splitting. The recall of minority classes such as chalk and marl collapsed to nearly zero "
    "under spatial extrapolation. A rejection threshold calibrated for a target accuracy of 90% "
    "was satisfied in all five random folds, but in only three of the fifteen cross-well and "
    "spatial folds. On continuous well sections, the boundary F1 score was only about 0.29 and "
    "the relative thickness error reached 41.5%. These results show that random depth-point "
    "validation markedly overstates field-deployment performance, and that a confidence "
    "threshold calibrated on the training domain cannot be transferred directly to new wells or "
    "new regions. Lithofacies studies should report spatial-extrapolation metrics, minority-class "
    "behaviour and continuous boundary accuracy, and low-confidence intervals should be referred "
    "for manual interpretation."
)
p = para(doc, "", indent_chars=0, space_after=2, line=1.3)
r = p.add_run("Abstract: ")
set_run(r, cn=SONG, en=TNR, size=9, bold=True)
r = p.add_run(abs_en)
set_run(r, cn=SONG, en=TNR, size=9)

p = para(doc, "", indent_chars=0, space_after=8, line=1.3)
r = p.add_run("Keywords: ")
set_run(r, cn=SONG, en=TNR, size=9, bold=True)
r = p.add_run("well logging; lithofacies classification; spatial cross-validation; "
              "selective prediction; confidence calibration; FORCE 2020")
set_run(r, cn=SONG, en=TNR, size=9)

# ================================================================ 0 引言
heading(doc, "0  引言", 1)
para(doc, "岩性识别是储层评价、地层对比与油气勘探开发中的一项基础工作。传统上，岩性主要由"
          "解释人员依据测井曲线形态、录井资料和岩心观察综合判定，过程耗时且高度依赖个人"
          "经验。随着测井数据规模的扩大，以随机森林、梯度提升决策树和人工神经网络为代表"
          "的机器学习方法被广泛用于测井岩性的自动识别（刘义等，2025）。FORCE 2020 测井岩性预测竞赛公开"
          "了北海地区百余口井的测井与岩性数据，推动了相关算法的快速发展，已有研究在该"
          "数据集上报告了较高的分类准确率（Bormann et al., 2020；Mousavi and Hosseini-Nasab, 2024）。")
para(doc, "然而，多数已有研究采用随机或分层抽样的交叉验证方式，默认各深度点相互独立。测井"
          "数据在深度方向上连续采样，相邻深度点的曲线值与岩性高度相关；同一构造单元内的"
          "邻井在岩性组合和测井响应上也存在空间相关。当训练集与测试集包含同一口井或空间"
          "邻近井的深度点时，模型实际上可借助这种相关性“记忆”局部模式，从而产生信息"
          "泄漏，使验证指标明显高于其在新井、新区域上的真实表现。在生态制图、滑坡易发性"
          "评价和土壤制图等领域，研究者已证明随机交叉验证会因空间自相关而系统性高估模型"
          "性能（Roberts et al., 2017；Kumar et al., 2025），空间分块交叉验证能够给出更接近实际的泛化估计；但在测井岩性识别中，对"
          "不同验证协议差异的系统比较仍不充分。")
para(doc, "另一方面，岩性识别往往服务于高风险的勘探决策，理想的预测系统不仅应给出岩性类别，"
          "还应在不确定时主动弃判（拒判），将低置信度井段交由人工复核，这一思想即选择性"
          "预测（Geifman and El-Yaniv, 2017）。已有选择性预测方法通常在与测试数据同分布的校准集上确定置信度阈值，其"
          "在分布变化、新井和新区域条件下能否保持预期精度，直接关系到该机制的实际可用性，"
          "但相关跨域可迁移性的检验较少。此外，逐点准确率难以反映模型对连续岩性层段边界"
          "位置和层厚的刻画能力，而后者正是测井解释更为关心的内容。")
para(doc, "针对上述问题，本文以公开的 FORCE 2020 数据集为对象，开展以下工作：①构建随机深度点、"
          "按井分组和带缓冲的空间分块三套交叉验证协议，量化验证协议对岩性识别评价结果的"
          "影响；②比较直方图梯度提升、随机森林和 LightGBM 三类模型，并结合多种子、多空间"
          "方案考察结论的稳定性；③通过嵌套校准的选择性预测，检验置信度拒判阈值向新井、"
          "新区域的可迁移性；④在连续井段上评价岩性边界定位与层厚估计精度，揭示逐点指标"
          "掩盖的问题，并讨论井距、地层组成差异与误差之间的关系。")

# ================================================================ 1 数据与研究区
heading(doc, "1  数据与研究区", 1)
heading(doc, "1.1  数据集来源与研究区概况", 2)
para(doc, "本文使用 FORCE 2020 测井岩性预测公开数据集，数据由挪威北海油气勘探相关机构整理，"
          "通过 Zenodo 平台发布，可公开获取。研究区位于挪威海域南、北维京地堑一带，钻井在"
          "平面上呈明显的群组分布。数据集共包含 118 口井的测井曲线、岩性解释标签及井位"
          "坐标，所有井均附有平面坐标信息。测井曲线涵盖井径、钻井液密度、机械钻速、多种"
          "电阻率、自然电位、声波时差、中子孔隙度、光电吸收截面指数、自然伽马、体积密度"
          "及密度校正值等常规测井项目，为岩性识别提供了较完整的特征基础。")
heading(doc, "1.2  岩性类别与样本筛选", 2)
para(doc, "数据集原始岩性标签包含 12 个类别。考虑到岩盐（Halite）仅出现在 4 口井、基底"
          "（Basement）仅出现在 1 口井，相关类别在按井和空间分块验证中无法被各折稳定覆盖，"
          "主分析将其排除并单独留档，最终保留 10 个岩性类别用于建模，包括页岩、砂岩、"
          "砂泥岩互层、石灰岩、泥灰岩、凝灰岩、白垩、煤、白云岩和硬石膏。数据集中存在"
          "相当比例的未标注深度，全部有标签深度点约 143.1 万个。各类别样本量差异显著，"
          "页岩等优势类别占比高，白云岩、煤等少数类别样本稀少，这种类别不平衡会对模型"
          "的逐类表现产生重要影响。")
heading(doc, "1.3  特征、井位与地层信息", 2)
para(doc, "本文选取 18 个数值型特征参与建模，包括测量深度、三维位置坐标（x、y、z）以及"
          "1.1 节所述的各条测井曲线。除测井响应外，数据集还提供井位平面坐标，并可结合"
          "挪威石油管理局公开的地层分层表建立地层组连接；官方地层组连接覆盖率约 99.2%，"
          "少量无法匹配的井段标记为未分配。井位坐标用于空间分块划分和制图，地层信息用于"
          "分析地层组成差异与预测误差之间的关系。需要说明的是，坐标和深度是否会造成"
          "额外的性能高估，将通过特征消融实验加以检验。")

# ================================================================ 2 研究方法
heading(doc, "2  研究方法", 1)
heading(doc, "2.1  三套交叉验证协议", 2)
para(doc, "为区分不同部署情景下的模型可靠性，本文设计三套由宽松到严格的五折交叉验证协议。")
para(doc, "（1）随机深度点划分。将全部采样深度点随机均分为五折，轮流以四折训练、一折测试。"
          "由于同一口井的相邻深度点可同时进入训练集和测试集，该协议不控制井内与空间相关，"
          "其结果代表一种乐观的性能上界，仅用于与其他协议对照，不能作为空间泛化能力的"
          "证据。")
para(doc, "（2）按井分组划分。以整口井为分组单位，将 118 口井分配到五折，测试折所含井的"
          "全部深度点均不参与训练，从而模拟模型在已具备区域其他井资料时对一口全新井的"
          "预测情景。")
para(doc, "（3）空间分块划分。依据各井的平面坐标进行聚类，将研究区划分为若干空间块，轮流"
          "留出一个完整块作为测试区，并在训练块与测试块之间设置缓冲区，剔除距测试块"
          "20 km 以内的井，使训练、测试井在空间上充分分离，用以模拟模型在一个全新地理"
          "区域的外推性能。主分析采用 5 块、20 km 缓冲方案；为检验结果对分块方式的"
          "敏感性，另设置 6 块、30 km 缓冲方案进行对照。")
heading(doc, "2.2  分类模型", 2)
para(doc, "本文选用三类在表格数据上表现稳健、且对缺失值和类别不平衡具有较好适应性的集成"
          "模型。直方图梯度提升（histogram gradient boosting）基于直方图近似的梯度提升"
          "决策树，通过迭代拟合残差并原生支持缺失值处理，学习率设为 0.08、最大迭代次数"
          "180、最大叶节点数 31，并启用早停。随机森林（random forest）通过自助采样和"
          "特征随机选取构建大量决策树并投票，采用 200 棵树、叶节点最小样本数 10、平衡"
          "子样本类别权重，并配合中位数缺失值填补。LightGBM 采用基于直方图的单侧采样与"
          "互斥特征捆绑策略，训练效率高，设置 350 棵树、学习率 0.05、叶节点数 31 和"
          "平衡类别权重。三类模型的其余参数保持默认或按统一设定，以保证协议间比较的"
          "公平性（Breiman, 2001；Friedman, 2001；Chen and Guestrin, 2016；Ke et al., 2017）。")
heading(doc, "2.3  评价指标", 2)
para(doc, "整体性能采用准确率、平衡准确率和宏平均 F1 评价。宏平均 F1 对十个类别等权平均，"
          "不受优势类别样本量主导，能更敏感地反映少数类的变化，是本文的主要比较指标。"
          "同时报告逐类召回率，以考察具体岩性类别的识别情况。在概率可靠性方面，采用"
          "期望校准误差（expected calibration error，ECE）衡量预测置信度与真实正确率"
          "的偏离，并以风险—覆盖曲线及其曲线下面积（AURC）刻画模型在不同弃判比例下的"
          "整体风险。")
heading(doc, "2.4  嵌套校准与选择性预测", 2)
para(doc, "选择性预测允许模型对置信度不足的样本弃判（El-Yaniv and Wiener, 2010）。对每个深度"
          "点，取模型输出的最大类别"
          "概率作为置信度；当置信度低于阈值时拒判，其余样本予以接受。随着阈值升高，被"
          "接受样本的比例（覆盖率）下降，而接受样本上的错误率（选择性风险）通常随之"
          "降低，二者构成风险—覆盖关系。为避免用测试数据选择阈值造成的信息泄漏，本文"
          "采用嵌套校准：在每一折内部，仅使用训练井数据再划分出独立的校准子集，在该"
          "子集上搜索使选择性精度达到目标值（90%）的最小置信度阈值，再将该阈值固定并"
          "应用于完全独立的测试折，统计测试折是否真正达到目标精度以及对应的覆盖率。"
          "若阈值在训练域校准成功却无法在测试域达标，即说明其跨域可迁移性不足。")
heading(doc, "2.5  连续井段边界与厚度评价", 2)
para(doc, "逐点指标无法体现岩性层段的连续性与边界定位能力。本文将预测结果按井和深度排序，"
          "以相邻深度间隔是否超过 0.5 m 为界划分连续测井段；在每个连续段内，将岩性类别"
          "发生变化的位置识别为岩性边界。对真实边界与预测边界，采用最近邻方式在 0.5 m "
          "容差内进行匹配，据此计算边界的精确率、召回率和 F1，并以匹配边界的深度偏差"
          "计算边界平均绝对误差。层厚方面，以相邻采样点的深度间隔为权重，分别累计各"
          "岩性类别的真实厚度与预测厚度，计算层厚平均绝对误差及其相对误差，同时统计"
          "预测层段数与真实层段数的偏差。上述指标可直接反映模型对岩性分界位置和单层"
          "厚度的刻画水平。")
heading(doc, "2.6  实验设计与数据采样", 2)
para(doc, "主分析在每口井内等间隔抽取至多 1000 个有标签深度点，以控制计算量并保证各井"
          "贡献均衡；在此基础上设置三个随机种子重复实验，结合两套空间分块方案，考察"
          "结论是否受单次划分或特定分块方式影响。连续井段评价则在完整的连续深度预测"
          "结果上进行，覆盖全部 118 口井。所有缺失值填补、类别权重处理、置信度校准和"
          "阈值选择均严格限定在对应训练折内部完成，测试数据不参与任何拟合与调参过程，"
          "以保证评价结果的无偏性。相关模型与数据处理基于 scikit-learn 实现（Pedregosa et al., 2011）。")

# ================================================================ 3 实验结果
heading(doc, "3  实验结果", 1)
heading(doc, "3.1  验证协议对整体性能的影响", 2)
para(doc, "表 1 给出直方图梯度提升模型在三套协议下的整体评价结果，图 1 进一步展示了各折宏"
          "平均 F1 的分布。可以看出，同一模型在不同验证协议下的性能差异十分显著：随机深度"
          "点划分下宏平均 F1 约为 0.79、准确率约 0.89，而按井分组划分宏平均 F1 降至约 0.42，"
          "空间分块划分进一步降至约 0.30。综合三个随机种子、共 15 次训练—测试的结果，"
          "直方图梯度提升的宏平均 F1 在随机、按井和空间分块（5 块、20 km）下分别为 0.788、"
          "0.424 和 0.301；改用 6 块、30 km 方案后空间分块为 0.292，结论不随分块方案改变。")
table_caption(doc, "表 1  直方图梯度提升模型在三套验证协议下的整体性能")
three_line_table(
    doc,
    ["验证协议", "准确率", "平衡准确率", "宏平均 F1", "期望校准误差"],
    [
        ["随机深度点", "0.891", "0.748", "0.790", "0.017"],
        ["按井分组", "0.738", "0.411", "0.424", "0.077"],
        ["空间分块", "0.688", "0.301", "0.300", "0.126"],
    ],
)
para(doc, "LightGBM 表现出完全一致的规律，其宏平均 F1 在随机、按井和空间分块下分别为 0.781、"
          "0.443 和 0.303；随机森林在试运行中对应值为 0.692、0.450 和 0.302。三类模型、"
          "多个随机种子和两套空间方案均表明：随机深度点验证显著高估了模型性能，一旦要求"
          "模型对全新井或全新区域外推，宏平均 F1 仅为随机划分时的约一半甚至更低。值得"
          "注意的是，空间分块下期望校准误差明显增大，说明模型在严格外推时置信度与真实"
          "正确率的一致性也在恶化。", space_before=4)
add_figure(doc, "02_protocol_macro_f1.png",
           "图 1  两类模型在三套验证协议下的宏平均 F1 箱线图")
heading(doc, "3.2  总体指标掩盖的少数类坍缩", 2)
para(doc, "图 2 以热力图形式给出各岩性类别在三套协议下的召回率。优势类别页岩在各协议下均"
          "保持 0.90 以上的高召回，砂岩、石灰岩等也相对稳定，这是总体准确率仍然不低的"
          "主要原因。然而，多个少数类在跨井和空间外推下出现严重坍缩：白垩的召回率由"
          "随机划分的 0.89 降至按井的 0.12、空间分块的 0.00；泥灰岩由 0.64 降至 0.18 和"
          "0.02；凝灰岩由 0.59 降至 0.04 和 0.02；白云岩在按井和空间分块下均为 0.00。"
          "这意味着模型在新区域实际上几乎无法识别这些岩性，而总体准确率因优势类别占比"
          "高而掩盖了这一问题。因此，仅报告准确率或加权 F1 会对岩性识别的可靠性形成"
          "误导，必须同时考察宏平均指标和逐类召回。")
add_figure(doc, "04_class_recall.png",
           "图 2  各岩性类别在三套验证协议下的召回率热力图")
heading(doc, "3.3  选择性预测的跨域可迁移性", 2)
para(doc, "图 3 给出三类协议下的风险—覆盖曲线。在随机深度点划分下，即使保留绝大部分深度点，"
          "选择性精度仍维持在较高水平；而按井和空间分块的曲线随覆盖率提高下降更快，说明"
          "在严格外推时，模型置信度对错误的区分能力减弱。为直接检验拒判阈值的可迁移性，"
          "以 90% 目标精度在训练折内嵌套校准阈值并应用于独立测试折，结果如图 4 所示："
          "随机深度点划分下五折全部达到 90% 目标精度，而按井分组仅 1 折达标，空间分块"
          "5 块方案 2 折、6 块方案 0 折达标，跨井与空间外推合计 15 折仅 3 折达标。")
add_figure(doc, "03_risk_coverage.png",
           "图 3  三类验证协议下的选择性风险—覆盖曲线")
add_figure(doc, "08_nested_selective.png",
           "图 4  嵌套校准下的覆盖率、选择性精度与校准阈值")
para(doc, "与此同时，达到目标精度所需的覆盖率也大幅下降。直方图梯度提升模型在随机划分下"
          "可保留约 91% 的深度点，而按井分组和空间分块下仅能保留约 42% 和 21%，即模型"
          "需要弃判近六成乃至八成的样本才能勉强维持精度，且多数情况下仍无法稳定达标。"
          "这表明，在训练井上校准得到的固定置信度阈值缺乏跨域可迁移性：在新井、新区域，"
          "同样的阈值既不能保证预期精度，也会造成过高的弃判比例。", space_before=4)
heading(doc, "3.4  连续井段的边界定位与层厚误差", 2)
para(doc, "表 2 汇总了连续井段上的边界与层厚评价结果，全部基于 118 口井的完整连续预测。"
          "尽管按井、空间协议下的逐点准确率仍分别有 0.74 和 0.69 左右，但岩性边界 F1 仅"
          "为 0.29～0.32，意味着仅有约三成的岩性分界能够在 0.5 m 容差内被正确定位。层厚"
          "估计的偏差更为突出：按井分组下层厚平均绝对误差约 81～87 m、相对误差约 33%～36%，"
          "空间分块下厚度平均绝对误差达 101～108 m、相对误差约 42%～46%。这说明模型虽然"
          "在多数深度点上给出了正确类别，却难以准确刻画岩性层段的分界位置和单层厚度，"
          "而这恰是测井解释和地层对比中最关键的信息，是逐点指标完全无法反映的。")
table_caption(doc, "表 2  连续井段岩性边界与层厚评价结果")
three_line_table(
    doc,
    ["协议", "模型", "点准确率", "边界 F1", "边界 MAE/m", "厚度 MAE/m", "厚度相对误差"],
    [
        ["按井分组", "直方图梯度提升", "0.739", "0.324", "0.233", "81.5", "0.332"],
        ["按井分组", "LightGBM", "0.705", "0.318", "0.232", "87.4", "0.360"],
        ["空间分块", "直方图梯度提升", "0.694", "0.290", "0.237", "101.0", "0.415"],
        ["空间分块", "LightGBM", "0.657", "0.290", "0.237", "107.6", "0.457"],
    ],
)
heading(doc, "3.5  特征消融分析", 2)
para(doc, "为判断随机划分下的高性能是否主要来自坐标、深度等位置信息的泄漏，设置四组特征集"
          "进行消融，结果如图 5 所示。在随机深度点划分下，全部 18 个特征的宏平均 F1 为 0.790，"
          "去除坐标后（测井曲线加深度）为 0.775，仅保留测井曲线时仍有 0.742；而在按井和"
          "空间协议下，无论是否包含坐标，宏平均 F1 都处于 0.29～0.42 的较低水平且差异很小。"
          "这说明位置特征虽有一定贡献，但随机划分与严格外推之间的巨大差距并不能简单归因于"
          "坐标泄漏，根本原因在于验证协议对井内、空间相关的控制；即使完全不使用坐标，模型"
          "在新区域的泛化能力依然有限。")
add_figure(doc, "09_feature_ablation.png",
           "图 5  四组特征集在三套验证协议下的宏平均 F1")
heading(doc, "3.6  井距与地层组成的误差归因", 2)
para(doc, "图 6 以散点形式展示各井到最近训练井的距离与该井准确率之间的关系。散点没有呈现"
          "稳定的单调趋势：距离较近的井既有高准确率也有低准确率，远距离井同样如此。进一步"
          "的相关分析表明，最近训练井距离与井级误差的相关关系较弱且方向不一致，多数情况下"
          "未通过显著性检验；以地层组、岩性组成差异度量的分布偏移与误差之间也仅存在弱相关。"
          "因此，本文不将井距或地层组成差异作为预测误差的单调代理。这一负结果提示，测井"
          "岩性外推误差受局部测井响应、标签质量、地层相变等多重因素共同影响，难以用单一"
          "距离变量解释。")
add_figure(doc, "06_distance_accuracy.png",
           "图 6  井级准确率与最近训练井距离的关系")

# ================================================================ 4 讨论
heading(doc, "4  讨论", 1)
heading(doc, "4.1  验证协议差异的本质", 2)
para(doc, "本文观察到的性能落差与空间自相关导致的信息泄漏密切相关。测井曲线在深度方向连续"
          "变化、岩性呈层状分布，相邻深度点并非独立样本；同一构造单元内的邻井也共享相似"
          "的沉积与地层特征。随机深度点划分使训练集与测试集包含大量高度相关的样本，模型"
          "可利用局部模式获得高分，这一机制与滑坡易发性、土壤制图等领域报道的随机交叉验证"
          "高估现象一致。按井分组切断了井内泄漏，空间分块进一步切断了井间的空间泄漏并"
          "引入缓冲（Valavi et al., 2019），因而更接近真实的外推部署。特征消融显示仅用测井曲线时随机与外推差距"
          "依然存在，说明问题的根源是验证协议而非单纯的坐标特征。这提示研究者在测井岩性"
          "研究中应将按井或空间分块验证作为默认评价方式，随机划分仅可作为乐观上界报告。")
heading(doc, "4.2  选择性预测跨域失效的原因", 2)
para(doc, "选择性预测的有效性依赖于置信度能够正确区分正确与错误样本，且校准集与部署场景"
          "分布一致。在新井、新区域，测井响应与岩性组合发生不同程度的分布变化，模型在"
          "训练域学到的概率分布出现校准漂移，空间外推下期望校准误差明显增大即反映了这一"
          "点。固定阈值本质上是针对训练域的特定分位数设定的，分布改变后，同一置信度水平"
          "对应的真实正确率随之改变，导致阈值既无法保证目标精度，又会产生过高弃判率。"
          "因此，跨域部署时不宜直接搬用训练域阈值，更可行的做法是结合目标区域少量有标签"
          "井重新校准，或采用对分布变化更稳健的置信度估计与自适应阈值方法（Niculescu-Mizil and Caruana, 2005；Guo et al., 2017；Geifman et al., 2019）。")
heading(doc, "4.3  连续井段评价的测井解释意义", 2)
para(doc, "岩性识别的最终用途是地层划分与对比，解释人员关心的是岩性分界的位置和各层厚度，"
          "而非孤立深度点的对错。本文结果显示，在逐点准确率尚可的情况下，边界 F1 仅约三成、"
          "层厚相对误差达三分之一以上，说明模型预测往往表现为层段内的零星错判和边界位置"
          "偏移，破坏了岩性序列的连续性。仅以逐点指标评价会对模型的实际解释价值形成高估。"
          "引入边界、层厚等面向层段的指标，有助于更客观地评估模型能否真正用于地层划分。")
para(doc, "为检验最小游程平滑这一常见后处理是否有效，本文在 1、3、5、10 m 四个最小游程"
          "阈值下重复全部连续井段评价。结果表明，平滑并未改善边界识别，反而使其恶化："
          "随着阈值增大，边界精确率有所上升，但边界召回率大幅下降，边界 F1 持续降低。"
          "以直方图梯度提升模型为例，按井分组的边界 F1 由未平滑的 0.324 降至 1 m 平滑的"
          "0.249 和 10 m 平滑的 0.060，空间分块下由 0.290 降至 0.222 和 0.057；点准确率"
          "同时轻微下降，预测层段数与真实层段数的平均绝对偏差高达 200～322 个。其原因"
          "在于，真实岩性序列包含大量薄层，平滑将短游程强行并入相邻层，在消除零星错判的"
          "同时也抹掉了大量真实分界，使召回率损失远超精确率增益。因此，最小游程平滑并不"
          "适合作为默认后处理；若要使用，必须结合目标区域的最小可采厚度，并在边界召回率"
          "约束下谨慎选择阈值。")
heading(doc, "4.4  局限性与展望", 2)
para(doc, "本文存在以下局限。其一，主分析为控制计算量对每井至多采样 1000 个深度点，虽经"
          "多种子验证，但与全量数据的结果仍可能存在差异，连续井段评价已尽量在完整预测上"
          "进行，后续可在全量数据上进一步复核。其二，井距与地层组成未能稳定解释误差，这一"
          "负结果意味着误差成因仍需结合测井质量、标签不确定性和局部相变深入分析，本文未"
          "强行建立简单因果关系。其三，模型范围限于三类树集成方法，未涵盖深度神经网络，"
          "但已有研究表明不同模型在严格外推下的表现趋势相近，本文结论的重点在于评价协议"
          "而非特定算法。未来可在全量数据上开展跨域重新校准、面向连续层段的损失函数设计"
          "以及新井与新地层双重外推等研究。")

# ================================================================ 5 结论
heading(doc, "5  结论", 1)
para(doc, "本文以公开的 FORCE 2020 数据集为对象，通过随机深度点、按井分组和空间分块三套"
          "验证协议，结合三类树集成模型、嵌套选择性预测和连续井段评价，系统评估了测井"
          "岩性识别的可靠性，主要结论如下。")
para(doc, "（1）验证协议对评价结果具有决定性影响。直方图梯度提升模型的宏平均 F1 由随机划分"
          "的 0.788 降至按井分组的 0.424 和空间分块的 0.301，该规律在三类模型、多个随机"
          "种子和两套空间方案下一致，表明随机深度点验证显著高估了模型的现场部署性能。")
para(doc, "（2）总体准确率掩盖了少数类坍缩。白垩、泥灰岩、凝灰岩和白云岩等少数类在空间"
          "外推下召回率降至接近零，而优势类别仍保持高召回，岩性识别评价必须同时报告宏"
          "平均指标和逐类召回。")
para(doc, "（3）训练域校准的拒判阈值缺乏跨域可迁移性。目标精度 90% 的阈值在随机划分下五折"
          "全部达标，而跨井与空间外推 15 折仅 3 折达标，且覆盖率由约 91% 降至约 42% 和 21%。")
para(doc, "（4）连续井段的岩性边界定位能力薄弱。边界 F1 仅约 0.29～0.32，层厚相对误差达"
          "33%～46%，这是逐点指标无法反映、却对测井解释至关重要的问题；最小游程平滑会"
          "因合并薄层而进一步降低边界 F1，不宜作为默认后处理。")
para(doc, "综上，测井岩性识别研究应以按井或空间分块验证为主，报告少数类与连续边界表现，"
          "并将低置信度井段交由人工复核；在新井、新区域部署时，应重新校准置信度阈值，"
          "而非直接搬用训练域结果。")

# 致谢
heading(doc, "致谢", 1)
para(doc, "感谢 FORCE 2020 竞赛及相关机构公开提供测井与岩性数据，使本研究得以开展。"
          "[基金项目名称及编号待补；其他需要致谢的单位或个人请在此补充。]")

# ================================================================ 参考文献
heading(doc, "参考文献", 1)


def ref(text):
    p = para(doc, text, cn=SONG, en=TNR, size=9, indent_chars=0,
             space_after=1, line=1.15)
    pf = p.paragraph_format
    pf.left_indent = Pt(18)
    pf.first_line_indent = Pt(-18)
    return p


# 中文文献（含原始英文对照）
ref("刘义, 胡修权, 李娜, 等, 2025. 机器学习在非常规油气藏岩相研究中的应用进展[J]. "
    "中国地质, 52(5): 1557-1575.")
ref("Liu Y, Hu X Q, Li N, et al., 2025. Application of machine learning in the "
    "lithofacies study of unconventional oil and gas reservoirs[J]. Geology in "
    "China, 52(5): 1557-1575. (In Chinese)")

# 外文文献（按第一作者姓氏字母排序）
ref("Bormann P, Aursand P, Dilib F, et al., 2020. FORCE 2020 well well log and "
    "lithofacies dataset for machine learning competition[Data set]. Zenodo. "
    "DOI: 10.5281/zenodo.4351156.")
ref("Breiman L, 2001. Random forests[J]. Machine Learning, 45(1): 5-32.")
ref("Chen T, Guestrin C, 2016. XGBoost: a scalable tree boosting system[C]// "
    "Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge "
    "Discovery and Data Mining: 785-794.")
ref("El-Yaniv R, Wiener Y, 2010. On the foundations of noise-free selective "
    "classification[J]. Journal of Machine Learning Research, 11: 1605-1641.")
ref("Friedman J H, 2001. Greedy function approximation: a gradient boosting "
    "machine[J]. The Annals of Statistics, 29(5): 1189-1232.")
ref("Geifman Y, El-Yaniv R, 2017. Selective classification for deep neural "
    "networks[C]// Advances in Neural Information Processing Systems 30.")
ref("Geifman Y, Uziel G, El-Yaniv R, 2019. Bias-reduced uncertainty estimation "
    "for deep neural classifiers[C]// International Conference on Learning "
    "Representations.")
ref("Guo C, Pleiss G, Sun Y, et al., 2017. On calibration of modern neural "
    "networks[C]// Proceedings of the 34th International Conference on Machine "
    "Learning: 1321-1330.")
ref("Ke G, Meng Q, Finley T, et al., 2017. LightGBM: a highly efficient "
    "gradient boosting decision tree[C]// Advances in Neural Information "
    "Processing Systems 30.")
ref("Kumar C, Walton G, Santi P, et al., 2025. Random cross-validation produces "
    "biased assessment of machine learning performance in regional landslide "
    "susceptibility prediction[J]. Remote Sensing, 17(2): 213. "
    "DOI: 10.3390/rs17020213.")
ref("Mousavi S H R, Hosseini-Nasab S M, 2024. A novel approach to classify "
    "lithology of reservoir formations using GrowNet and Deep-Insight with "
    "physic-based feature augmentation[J]. Energy Science & Engineering. "
    "DOI: 10.1002/ese3.1895.")
ref("Niculescu-Mizil A, Caruana R, 2005. Predicting good probabilities with "
    "supervised learning[C]// Proceedings of the 22nd International Conference "
    "on Machine Learning: 625-632.")
ref("Pedregosa F, Varoquaux G, Gramfort A, et al., 2011. Scikit-learn: machine "
    "learning in Python[J]. Journal of Machine Learning Research, 12: "
    "2825-2830.")
ref("Roberts D R, Bahn V, Ciuti S, et al., 2017. Cross-validation strategies "
    "for data with temporal, spatial, hierarchical, or phylogenetic "
    "structure[J]. Ecography, 40(8): 913-929.")
ref("Valavi R, Elith J, Lahoz-Monfort J J, et al., 2019. blockCV: an R package "
    "for generating spatially or environmentally separated folds for k-fold "
    "cross-validation of species distribution models[J]. Methods in Ecology "
    "and Evolution, 10(2): 225-232.")

# 保存
OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(str(OUT))
print("saved:", OUT)









