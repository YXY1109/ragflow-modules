import copy
import re

import chardet
from PIL import Image

from nlp import rag_tokenizer

all_codecs = [
    'utf-8', 'gb2312', 'gbk', 'utf_16', 'ascii', 'big5', 'big5hkscs',
    'cp037', 'cp273', 'cp424', 'cp437',
    'cp500', 'cp720', 'cp737', 'cp775', 'cp850', 'cp852', 'cp855', 'cp856', 'cp857',
    'cp858', 'cp860', 'cp861', 'cp862', 'cp863', 'cp864', 'cp865', 'cp866', 'cp869',
    'cp874', 'cp875', 'cp932', 'cp949', 'cp950', 'cp1006', 'cp1026', 'cp1125',
    'cp1140', 'cp1250', 'cp1251', 'cp1252', 'cp1253', 'cp1254', 'cp1255', 'cp1256',
    'cp1257', 'cp1258', 'euc_jp', 'euc_jis_2004', 'euc_jisx0213', 'euc_kr',
    'gb18030', 'hz', 'iso2022_jp', 'iso2022_jp_1', 'iso2022_jp_2',
    'iso2022_jp_2004', 'iso2022_jp_3', 'iso2022_jp_ext', 'iso2022_kr', 'latin_1',
    'iso8859_2', 'iso8859_3', 'iso8859_4', 'iso8859_5', 'iso8859_6', 'iso8859_7',
    'iso8859_8', 'iso8859_9', 'iso8859_10', 'iso8859_11', 'iso8859_13',
    'iso8859_14', 'iso8859_15', 'iso8859_16', 'johab', 'koi8_r', 'koi8_t', 'koi8_u',
    'kz1048', 'mac_cyrillic', 'mac_greek', 'mac_iceland', 'mac_latin2', 'mac_roman',
    'mac_turkish', 'ptcp154', 'shift_jis', 'shift_jis_2004', 'shift_jisx0213',
    'utf_32', 'utf_32_be', 'utf_32_le', 'utf_16_be', 'utf_16_le', 'utf_7', 'windows-1250', 'windows-1251',
    'windows-1252', 'windows-1253', 'windows-1254', 'windows-1255', 'windows-1256',
    'windows-1257', 'windows-1258', 'latin-2'
]


def find_codec(blob):
    detected = chardet.detect(blob[:1024])
    if detected['confidence'] > 0.5:
        if detected['encoding'] == "ascii":
            return "utf-8"

    for c in all_codecs:
        try:
            blob[:1024].decode(c)
            return c
        except Exception:
            pass
        try:
            blob.decode(c)
            return c
        except Exception:
            pass

    return "utf-8"


def concat_img(img1, img2):
    if img1 and not img2:
        return img1
    if not img1 and img2:
        return img2
    if not img1 and not img2:
        return None

    if img1 is img2:
        return img1

    if isinstance(img1, Image.Image) and isinstance(img2, Image.Image):
        pixel_data1 = img1.tobytes()
        pixel_data2 = img2.tobytes()
        if pixel_data1 == pixel_data2:
            return img1

    width1, height1 = img1.size
    width2, height2 = img2.size

    new_width = max(width1, width2)
    new_height = height1 + height2
    new_image = Image.new('RGB', (new_width, new_height))

    new_image.paste(img1, (0, 0))
    new_image.paste(img2, (0, height1))
    return new_image


def tokenize(d, t, eng):
    d["content_with_weight"] = t
    t = re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", t)
    d["content_ltks"] = rag_tokenizer.tokenize(t)
    d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])


def add_positions(d, poss):
    if not poss:
        return
    page_num_int = []
    position_int = []
    top_int = []
    for pn, left, right, top, bottom in poss:
        page_num_int.append(int(pn + 1))
        top_int.append(int(top))
        position_int.append((int(pn + 1), int(left), int(right), int(top), int(bottom)))
    d["page_num_int"] = page_num_int
    d["position_int"] = position_int
    d["top_int"] = top_int


def tokenize_table(tbls, doc, eng, batch_size=10):
    res = []
    # add tables
    for (img, rows), poss in tbls:
        if not rows:
            continue
        if isinstance(rows, str):
            d = copy.deepcopy(doc)
            tokenize(d, rows, eng)
            d["content_with_weight"] = rows
            if img:
                d["image"] = img
                d["doc_type_kwd"] = "image"
            if poss:
                add_positions(d, poss)
            res.append(d)
            continue
        de = "; " if eng else "； "
        for i in range(0, len(rows), batch_size):
            d = copy.deepcopy(doc)
            r = de.join(rows[i:i + batch_size])
            tokenize(d, r, eng)
            if img:
                d["image"] = img
                d["doc_type_kwd"] = "image"
            add_positions(d, poss)
            res.append(d)
    return res
