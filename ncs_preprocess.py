import json
import re

# 1. ncs_colors.json 파일을 같은 폴더에 넣고, 경로가 다르면 경로만 수정
with open("ncs_colors.json", "r", encoding="utf-8") as f:
    data = json.load(f)

result = []
for color in data:
    ncs_name = color["name"]
    hex_code = color["hex"]
    # NCS S 2030-Y50R → 20: 흑색도, 30: 순색도, 백색도=100-흑색도-순색도
    match = re.match(r"NCS S (\d{2})(\d{2})", ncs_name)
    if match:
        blackness = int(match.group(1))
        chromaticness = int(match.group(2))
        whiteness = 100 - blackness - chromaticness
    else:
        # 무채색 or 예외: S 0500-N, S 1000-N 등
        match_gray = re.match(r"NCS S (\d{2})(00)-N", ncs_name)
        if match_gray:
            blackness = int(match_gray.group(1))
            chromaticness = 0
            whiteness = 100 - blackness
        else:
            blackness = chromaticness = whiteness = None

    result.append({
        "name": ncs_name,
        "hex": hex_code,
        "blackness": blackness,
        "whiteness": whiteness,
        "chromaticness": chromaticness
    })

with open("ncs_colors_with_bw.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print("완료! ncs_colors_with_bw.json 파일이 생성됨.")
