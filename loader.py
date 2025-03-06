import os
import cv2
import numpy as np
import pytesseract
import torch
from transformers import AutoTokenizer
from PIL import Image
import itertools

max_seq_length = 512
max_connections = 100

def getitem_geo(img: Image, tokenizer: AutoTokenizer):
    # Perform OCR with detailed output
    ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    # Extract words, confidence, and bounding boxes
    words = ocr_data["text"]
    confidences = ocr_data["conf"]
    block = ocr_data["block_num"]
    para = ocr_data["par_num"]
    # bounding_boxes = list(
    #     zip(ocr_data["left"], ocr_data["top"], ocr_data["width"], ocr_data["height"])
    # )
    bounding_boxes = [[[left, top],
           [left + width, top],
           [left + width, top + height],
           [left, top + height]
        ] for left, top, width, height in zip(ocr_data["left"], ocr_data["top"], ocr_data["width"], ocr_data["height"])]
    
    # Filter out empty words
    processed_data = [
        {"word": word, "confidence": conf, "boundingBox": bbox, "tokens": tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))}
        for word, conf, bbox in zip(words, confidences, bounding_boxes)
        if word.strip() #returns stripped string
    ]
    processed_data['blocks']['first_token_idx_list'] = []
    processed_data['blocks']['boxes'] = []
    return_dict = {}

    width, height = img.size
    # width = json_obj["meta"]["imageSize"]["width"]
    # height = json_obj["meta"]["imageSize"]["height"]

    # img_path = os.path.join(dataset_root_path, json_obj["meta"]["image_path"])

    image = cv2.resize(img, (img_w, img_h))
    # image = cv2.resize(cv2.imread(img_path, 1), (img_w, img_h))
    image = image.astype("float32").transpose(2, 0, 1)

    pad_token_id = tokenizer.vocab["[PAD]"]
    cls_token_id = tokenizer.vocab["[CLS]"]
    sep_token_id = tokenizer.vocab["[SEP]"]
    unk_token_id = tokenizer.vocab["[UNK]"]

    # return_dict["image_path"] = img_path
    return_dict["image"] = image
    return_dict["size_raw"] = np.array([width, height])

    return_dict["input_ids"] = np.ones(max_seq_length, dtype=int) * pad_token_id
    return_dict["bbox_4p_normalized"] = np.zeros((max_seq_length, 8), dtype=np.float32)
    return_dict["attention_mask"] = np.zeros(max_seq_length, dtype=int)
    return_dict["first_token_idxes"] = np.zeros(max_block_num, dtype=int)
    return_dict["block_mask"] = np.zeros(max_block_num, dtype=int)
    return_dict["bbox"] = np.zeros((max_seq_length, 4), dtype=np.float32)
    return_dict["line_rank_id"] = np.zeros(max_seq_length, dtype=int)
    return_dict["line_rank_inner_id"] = np.ones(max_seq_length, dtype=int)

    list_tokens = []
    list_bbs = [] # word boxes
    list_blk_bbs = [] # block boxes
    box2token_span_map = []

    box_to_token_indices = []
    cum_token_idx = 0

    cls_bbs = [0.0] * 8
    cls_bbs_blk = [0] * 4

    token_indices_to_wordidx = [-1] * max_seq_length

    for word_idx, word in enumerate(processed_data):
        this_box_token_indices = []

        tokens = word["tokens"]
        bb = word["boundingBox"]
        if len(tokens) == 0:
            tokens.append(unk_token_id)

        if len(list_tokens) + len(tokens) > max_seq_length - 2:
            break # truncation for long documents

        box2token_span_map.append(
            [len(list_tokens) + 1, len(list_tokens) + len(tokens) + 1]
        )  # including st_idx, start from 1
        list_tokens += tokens

        # min, max clipping
        for coord_idx in range(4):
            bb[coord_idx][0] = max(0.0, min(bb[coord_idx][0], width))
            bb[coord_idx][1] = max(0.0, min(bb[coord_idx][1], height))

        bb = list(itertools.chain(*bb))
        bbs = [bb for _ in range(len(tokens))]

        for _ in tokens:
            cum_token_idx += 1
            this_box_token_indices.append(cum_token_idx) # start from 1
            token_indices_to_wordidx[cum_token_idx] = word_idx

        list_bbs.extend(bbs)
        box_to_token_indices.append(this_box_token_indices)

    sep_bbs = [width, height] * 4
    sep_bbs_blk = [width, height] * 2

    first_token_idx_list = processed_data['blocks']['first_token_idx_list'][:max_block_num]
    if first_token_idx_list[-1] > len(list_tokens):
        blk_length = max_block_num
        for blk_id, first_token_idx in enumerate(first_token_idx_list):
            if first_token_idx > len(list_tokens):
                blk_length = blk_id
                break
        first_token_idx_list = first_token_idx_list[:blk_length]
        
    first_token_ext = first_token_idx_list + [len(list_tokens) + 1]
    line_id = 1
    for blk_idx in range(len(first_token_ext) - 1):
        token_span = first_token_ext[blk_idx+1] - first_token_ext[blk_idx]
        # block box
        bb_blk = processed_data['blocks']['boxes'][blk_idx]
        bb_blk[0] = max(0, min(bb_blk[0], width))
        bb_blk[1] = max(0, min(bb_blk[1], height))
        bb_blk[2] = max(0, min(bb_blk[2], width))
        bb_blk[3] = max(0, min(bb_blk[3], height))
        list_blk_bbs.extend([bb_blk for _ in range(token_span)])
        # line_rank_id
        return_dict["line_rank_id"][first_token_ext[blk_idx]:first_token_ext[blk_idx+1]] = line_id
        line_id += 1
        # line_rank_inner_id
        if token_span > 1:
            return_dict["line_rank_inner_id"][first_token_ext[blk_idx]:first_token_ext[blk_idx+1]] = [1] + [2] * (token_span - 2) + [3]

    # For [CLS] and [SEP]
    list_tokens = (
        [cls_token_id]
        + list_tokens[: max_seq_length - 2]
        + [sep_token_id]
    )
    if len(list_bbs) == 0:
        # When len(json_obj["words"]) == 0 (no OCR result)
        list_bbs = [cls_bbs] + [sep_bbs]
        list_blk_bbs = [cls_bbs_blk] + [sep_bbs_blk]
    else:  # len(list_bbs) > 0
        list_bbs = [cls_bbs] + list_bbs[: max_seq_length - 2] + [sep_bbs]
        list_blk_bbs = [cls_bbs_blk] + list_blk_bbs[: max_seq_length - 2] + [sep_bbs_blk]

    len_list_tokens = len(list_tokens)
    len_blocks = len(first_token_idx_list)
    return_dict["input_ids"][:len_list_tokens] = list_tokens
    return_dict["attention_mask"][:len_list_tokens] = 1
    return_dict["first_token_idxes"][:len(first_token_idx_list)] = first_token_idx_list
    return_dict["block_mask"][:len_blocks] = 1
    return_dict["line_rank_inner_id"] = return_dict["line_rank_inner_id"] * return_dict["attention_mask"]

    bbox_4p_normalized = return_dict["bbox_4p_normalized"]
    bbox_4p_normalized[:len_list_tokens, :] = list_bbs

    # bounding box normalization -> [0, 1]
    bbox_4p_normalized[:, [0, 2, 4, 6]] = bbox_4p_normalized[:, [0, 2, 4, 6]] / width
    bbox_4p_normalized[:, [1, 3, 5, 7]] = bbox_4p_normalized[:, [1, 3, 5, 7]] / height

    if backbone_type == "layoutlm":
        bbox_4p_normalized = bbox_4p_normalized[:, [0, 1, 4, 5]]
        bbox_4p_normalized = bbox_4p_normalized * 1000
        bbox_4p_normalized = bbox_4p_normalized.astype(int)

    return_dict["bbox_4p_normalized"] = bbox_4p_normalized
    bbox = return_dict["bbox"]

    bbox[:len_list_tokens, :] = list_blk_bbs
    # bbox -> [0, 1000)
    bbox[:, [0, 2]] = bbox[:, [0, 2]] / width * 1000
    bbox[:, [1, 3]] = bbox[:, [1, 3]] / height * 1000
    bbox = bbox.astype(int)
    return_dict["bbox"] = bbox

    st_indices = [
        indices[0]
        for indices in box_to_token_indices
        if indices[0] < max_seq_length
    ]
    return_dict["are_box_first_tokens"][st_indices] = True


    for k in return_dict.keys():
        if isinstance(return_dict[k], np.ndarray):
            return_dict[k] = torch.from_numpy(return_dict[k])

    return return_dict