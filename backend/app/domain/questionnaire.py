from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    id: str
    module: str
    title: str
    key: str
    options: list[str]
    multi: bool = False
    optional: bool = False


TEN_SECTIONS: list[dict] = []

_MODULES: list[dict] = [
    {
        "module": "寒热",
        "questions": [
            {"id": "cold_heat", "title": "最近怕冷还是怕热更明显？", "key": "cold_heat", "options": ["明显怕冷", "怕冷为主", "无明显寒热", "怕热为主", "明显怕热", "忽冷忽热", "午后或夜间潮热"]},
            {"id": "limbs", "title": "手脚温度如何？", "key": "limbs", "options": ["手脚冰凉", "手脚偏凉", "手脚温和平稳", "手心脚心发热", "手足心热且烦", "说不清"]},
            {"id": "drink_temp", "title": "喝东西偏好哪种温度？", "key": "drink_temp", "options": ["喜热饮", "偏好温热", "冷热均可", "偏好凉爽", "喜冷饮", "口中淡不渴"]},
            {"id": "thirst", "title": "口渴情况怎样？", "key": "thirst", "options": ["口不渴", "轻微口渴", "明显口渴喜饮", "口渴但不想多喝", "夜间口干明显", "口黏不想喝水"]},
        ],
    },
    {
        "module": "汗",
        "questions": [
            {"id": "sweat", "title": "出汗情况如何？", "key": "sweat", "options": ["几乎不出汗", "出汗正常", "稍活动即出汗（自汗）", "入睡后盗汗", "白天动则汗出夜间亦汗", "出汗很少且怕风"]},
            {"id": "sweat_site", "title": "出汗多在哪个部位？", "key": "sweat_site", "options": ["全身", "头部为主", "手足心", "心胸部位", "半身", "不太清楚"], "optional": True},
        ],
    },
    {
        "module": "疼痛",
        "questions": [
            {"id": "pain_present", "title": "身体有明显疼痛或不适吗？", "key": "pain_present", "options": ["没有明显疼痛", "偶尔疼痛", "经常疼痛", "持续疼痛"]},
            {"id": "pain_nature", "title": "若有疼痛，性质更接近哪种？", "key": "pain_nature", "options": ["隐痛绵绵", "胀痛", "刺痛固定", "走窜作痛", "冷痛喜温", "重着酸痛", "灼热疼痛"], "optional": True},
            {"id": "pain_relief", "title": "怎样会舒服一些？", "key": "pain_relief", "options": ["按压或热敷后舒服", "活动后减轻", "休息后减轻", "遇冷加重", "遇热加重", "无明显缓解方式"], "optional": True},
        ],
    },
    {
        "module": "饮食与口味",
        "questions": [
            {"id": "appetite", "title": "食欲如何？", "key": "appetite", "options": ["食欲正常", "食欲下降", "食欲很差不想吃", "容易饥饿", "饿但不想吃", "饭后胀满"]},
            {"id": "taste", "title": "口中滋味有异常吗？", "key": "taste", "options": ["正常", "口淡无味", "口甜黏腻", "口苦", "口酸", "口咸", "口中泛恶"], "optional": True},
            {"id": "bloating", "title": "吃完后腹部感觉？", "key": "bloating", "options": ["无明显不适", "容易腹胀", "胃脘胀满", "嗳气打嗝", "矢气多", "食后困倦"]},
        ],
    },
    {
        "module": "二便",
        "questions": [
            {"id": "stool", "title": "大便情况？", "key": "stool", "options": ["每日一次成形", "大便偏干", "大便稀溏", "先干后溏", "泻下急迫臭秽", "大便不成形时溏时干", "数日一行", "便后仍有排不尽感"]},
            {"id": "stool_freq", "title": "大便频次？", "key": "stool_freq", "options": ["每日一至两次", "每日多次", "两三天一次", "一天数次且稀", "不规律"], "optional": True},
            {"id": "urine", "title": "小便情况？", "key": "urine", "options": ["正常", "小便清长量多", "小便黄短", "夜尿频繁", "尿频", "排尿不畅或灼热"]},
        ],
    },
    {
        "module": "睡眠与精神",
        "questions": [
            {"id": "sleep", "title": "睡眠如何？", "key": "sleep", "options": ["睡眠安稳", "入睡困难", "睡后易醒", "多梦纷纭", "早醒后难再睡", "嗜睡困倦", "醒后疲乏"]},
            {"id": "energy", "title": "精神体力状态？", "key": "energy", "options": ["精力充沛", "容易疲倦", "倦怠懒言", "乏力动则气短", "烦躁难以放松", "精神萎靡"]},
        ],
    },
    {
        "module": "胸腹与头身",
        "questions": [
            {"id": "chest", "title": "胸胁或心胸部感觉？", "key": "chest", "options": ["无不适", "胸闷", "心悸心慌", "胸胁胀满", "胸痛", "气息短促"], "optional": True},
            {"id": "dizziness", "title": "头晕头痛情况？", "key": "dizziness", "options": ["无", "经常头晕", "偶发头晕", "头重如裹", "头痛", "眩晕耳鸣"], "optional": True},
            {"id": "body_weight", "title": "身体沉重或浮肿感？", "key": "body_weight", "options": ["无", "身体困重", "四肢沉重", "晨起面部或眼睑浮肿", "下肢按压有凹陷感", "关节沉重酸痛"], "optional": True},
        ],
    },
    {
        "module": "情志",
        "questions": [
            {"id": "mood", "title": "近期情绪状态？", "key": "mood", "options": ["平稳", "情绪低落", "焦虑紧张", "急躁易怒", "多思多虑", "受惊易恐"]},
            {"id": "sigh", "title": "是否常叹气或觉得气不顺？", "key": "sigh", "options": ["不会", "偶尔叹气", "经常叹气", "咽中如有物梗阻"], "optional": True},
        ],
    },
    {
        "module": "妇科与男科",
        "questions": [
            {"id": "gender", "title": "生理性别（用于专科问诊）", "key": "gender", "options": ["男", "女"]},
            {"id": "women_cycle", "title": "月经情况（女性填写）", "key": "women_cycle", "options": ["不适用", "周期规律量适中", "量多色鲜", "量少色淡", "量少色暗有块", "经期腹痛", "周期提前", "周期推后", "经期乳房胀痛"], "optional": True},
            {"id": "men_kidney", "title": "腰膝与精力（男性可填）", "key": "men_kidney", "options": ["不适用", "腰膝酸软", "腰酸怕冷", "性欲减退", "遗精早泄", "精力尚可"], "optional": True},
        ],
    },
    {
        "module": "旧病与体质",
        "questions": [
            {"id": "age", "title": "年龄段", "key": "age", "options": ["未成年", "青年", "中年", "老年"]},
            {"id": "chronic", "title": "是否有慢性基础疾病？", "key": "chronic", "options": ["无", "高血压", "糖尿病", "心脏病", "肝肾功能异常", "甲状腺疾病", "肿瘤相关", "其他长期用药"]},
            {"id": "pregnant", "title": "是否处于妊娠期或哺乳期？", "key": "pregnant", "options": ["否", "是", "不确定"]},
            {"id": "allergy", "title": "过敏史", "key": "allergy", "options": ["无明显过敏", "药物过敏", "食物过敏", "花粉等过敏", "不确定"]},
        ],
    },
]

_UNCERTAIN = "不确定"


def _to_dict() -> list[dict]:
    modules = []
    for section in _MODULES:
        questions = []
        for q in section["questions"]:
            options = list(q["options"])
            if _UNCERTAIN not in options:
                options.append(_UNCERTAIN)
            questions.append({
                "id": q["id"],
                "title": q["title"],
                "key": q["key"],
                "options": options,
                "optional": bool(q.get("optional")),
            })
        modules.append({"module": section["module"], "questions": questions})
    return modules


QUESTIONNAIRE: list[dict] = _to_dict()
TOTAL_QUESTIONS = sum(len(m["questions"]) for m in QUESTIONNAIRE)
