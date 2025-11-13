import os
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from collections import OrderedDict

from api.BRIDGE import BRIDGE_Model
import TransformerEncoder
from config import (
    PRETRAINED_WEIGHTS, CHECKPOINT_PATH, tokenizer,
    PAD_INDEX, CLS_INDEX, SEP_INDEX, MSK_INDEX, SEED,
    SINGLE_ANSWER_LENGTH, BEAM_SIZE, REV_NUM
)

class GlobalModel:
    model = None
    tokenizer = None
    device = None

    pad_i = PAD_INDEX
    sep_i = SEP_INDEX
    single_answer_length = SINGLE_ANSWER_LENGTH
    beam_size = BEAM_SIZE
    rev_num = REV_NUM

def load_model():
    print("正在加载模型和分词器...")
    GlobalModel.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GlobalModel.tokenizer = tokenizer
    
    pretrained_config = AutoConfig.from_pretrained(
        PRETRAINED_WEIGHTS, output_hidden_states=True
    )
    pretrained_encoder = AutoModel.from_pretrained(
        PRETRAINED_WEIGHTS, config=pretrained_config
    )

    GlobalModel.model = BRIDGE_Model(
        pretrained_encoder,
        pretrained_config,
        CLS_INDEX,
        SEP_INDEX,
        PAD_INDEX,
        MSK_INDEX,
        SEED
    )

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"未找到模型权重文件: {CHECKPOINT_PATH}")

    print(f"正在从 {CHECKPOINT_PATH} 加载权重...")
    state_dict = torch.load(CHECKPOINT_PATH, map_location=GlobalModel.device)

    if next(iter(state_dict)).startswith("module."):
        print("检测到 DataParallel 'module.' 前缀，正在移除...")
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        GlobalModel.model.load_state_dict(new_state_dict)
    else:
        GlobalModel.model.load_state_dict(state_dict)

    GlobalModel.model.to(GlobalModel.device)
    GlobalModel.model.eval()
    
    print(f"模型已成功加载到 {GlobalModel.device}！")

def predict(load_model, batch_data, single_answer_length, sep_idx, beam_size):
    load_model.eval()
    with torch.no_grad():
        que, rev, ans = batch_data
        outputs = []
        for k in range(que.size(0)):
            top_candidates = []
            seen_words = {}

            q = que[k:k + 1, :]
            r = rev[k:k + 1, :, :]
            output = torch.tensor([[[]]], dtype=torch.long).to(GlobalModel.device)

            _, output_ts = load_model((q, r, output))
            output_tbv, output_tbi = torch.topk(F.log_softmax(output_ts, dim=1), beam_size, dim=1)
            for p in range(beam_size):
                top_candidates.append(([output_tbi[:, p]], output_tbv[:, p]))
                seen_words[p] = {int(output_tbi[:, p].item()): 1}
            for t in range(single_answer_length - 1):
                candidates = {}
                for i in range(beam_size):
                    current_tokens = top_candidates[i][0]
                    if int(current_tokens[-1].item()) == sep_idx:
                        pass
                    else:
                        output_c = torch.tensor(current_tokens).unsqueeze(0).unsqueeze(0).to(GlobalModel.device)
                        _, output_cs = load_model((q, r, output_c))
                        output_cbv, output_cbi = torch.topk(F.log_softmax(output_cs, dim=1), beam_size, dim=1)
                        for j in range(beam_size):
                            current_p = output_cbi[:, j]
                            current_p_i = int(current_p.item())
                            current_p_c = seen_words[i].get(current_p_i, 0)
                            if current_p != current_tokens[-1] and current_p_c <= 1:
                                seen_words[i][current_p_i] = current_p_c + 1
                                beam_score = (top_candidates[i][1] * len(current_tokens) + output_cbv[:, j]) / \
                                             ((len(current_tokens) + 1) ** 0.7)
                                candidates[(i, j)] = (current_tokens + [current_p], beam_score)
                            else:
                                pass
                if len(candidates) == 0:
                    break
                top_beams = sorted(candidates.items(), key=lambda x: -x[1][1])[:beam_size]
                for top_beam_i, top_beam in enumerate(top_beams):
                    top_candidates[top_beam_i] = top_beam[1]
            
            final_tokens = [t.item() for t in top_candidates[0][0]]
            outputs.append(final_tokens)
        return outputs

def run_my_model_inference(prompt: str) -> str:
    model = GlobalModel.model
    tokenizer = GlobalModel.tokenizer
    device = GlobalModel.device
    pad_i = GlobalModel.pad_i
    sep_i = GlobalModel.sep_i
    rev_num = GlobalModel.rev_num
    beam_size = GlobalModel.beam_size
    ans_len = GlobalModel.single_answer_length

    q_tensor = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=50
    ).input_ids.to(device)

    r_tensor = torch.full(
        (1, rev_num, 1),
        pad_i,
        dtype=torch.long,
        device=device
    )

    a_tensor = torch.full((1, 1, 1), pad_i, dtype=torch.long, device=device)

    batch = (q_tensor, r_tensor, a_tensor)

    print(f"正在为 prompt 运行推理: {prompt}")
    prediction_token_ids_list = predict(
        model,
        batch,
        ans_len,
        sep_i,
        beam_size
    )
    
    prediction_ids = prediction_token_ids_list[0]

    response_text = tokenizer.decode(
        prediction_ids,
        skip_special_tokens=True
    )
    
    print(f"模型生成: {response_text}")
    return response_text