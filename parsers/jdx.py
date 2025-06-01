import numpy as np
import re

def parse_jdx(file_path: str) -> dict[np.ndarray, np.ndarray]:
    with open(file_path, 'r', encoding='latin1') as f:
        lines = f.readlines()

    x_vals = []
    y_vals = []
    delta_x = None
    x_factor = 1.0
    y_factor = 1.0
    reading_data = False

    for line in lines:
        line = line.strip()
        if line.startswith("##DELTAX="):
            delta_x = float(line.split("=", 1)[1])
        elif line.startswith("##XFACTOR="):
            x_factor = float(line.split("=", 1)[1])
            continue
        elif line.startswith("##YFACTOR="):
            y_factor = float(line.split("=", 1)[1])
            continue
        elif line.startswith("##XYDATA="):
            reading_data = True
            continue
        elif line.startswith("##END="):
            break
        elif line.startswith("##"):
            continue
        elif reading_data:
            raw_tokens = re.split(r"(\s+|-)", line)
            tokens = []
            make_negative = False
            for token in raw_tokens:
                token = token.strip()
                if not token:
                    continue
                if token == '-':
                    make_negative = True
                elif token not in ('-', ''):
                    num = float(token)
                    if make_negative:
                        num = -num
                        make_negative = False
                    tokens.append(num)

            base_x = tokens[0]
            for i, y_str in enumerate(tokens[1:]):
                x_vals.append(base_x + i * delta_x)
                y_vals.append(float(y_str))

    x_scaled = np.array(x_vals) * x_factor
    y_scaled = np.array(y_vals) * y_factor
    return {'spectrum_x': x_scaled, 'spectrum_y': y_scaled}