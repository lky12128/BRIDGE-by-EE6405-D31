# config.py
from transformers import AutoTokenizer

PRETRAINED_WEIGHTS = "bert-base-cased"

CHECKPOINT_PATH = r"D:\OneDrive\Desktop\nlp_api\models\bert-base-cased_Pet_supplies_10R_1A_1.0_2.pth"

print("Config: 正在加载 Tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_WEIGHTS)
except Exception as e:
    print(f"加载 Tokenizer 失败: {e}")
    print("请确保 PRETRAINED_WEIGHTS 配置正确（示例：'bert-base-cased'）")
    exit()

PAD_INDEX = tokenizer.pad_token_id
CLS_INDEX = tokenizer.cls_token_id
SEP_INDEX = tokenizer.sep_token_id
MSK_INDEX = tokenizer.mask_token_id

if any(v is None for v in [PAD_INDEX, CLS_INDEX, SEP_INDEX, MSK_INDEX]):
    print("错误: Tokenizer 未能正确加载特殊 Token（pad/cls/sep/msk）")
    print(f"当前值 - PAD: {PAD_INDEX}, CLS: {CLS_INDEX}, SEP: {SEP_INDEX}, MSK: {MSK_INDEX}")
    print("请检查模型名称是否正确，已自动填充默认值")
    # 手动填充默认值（针对部分模型无默认特殊Token的情况）
    if PAD_INDEX is None:
        PAD_INDEX = 0
    if CLS_INDEX is None:
        CLS_INDEX = 1
    if SEP_INDEX is None:
        SEP_INDEX = 2
    if MSK_INDEX is None:
        MSK_INDEX = 3


SINGLE_ANSWER_LENGTH = 100000
BEAM_SIZE = 3
REV_NUM = 10
SEED = 123

print("Config: 配置加载完毕。")