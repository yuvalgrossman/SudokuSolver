import base64
import io
import cv2

import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
from copy import deepcopy
import json
from torchvision.transforms import ToTensor

from PIL import Image
from flask import Flask, request, render_template
from flask_cors import CORS

from detector.sudoku_detector import MultiClassifier

app = Flask(__name__)
CORS(app)

@app.route("/solve", methods=["POST"])
def solve():
    data = request.get_json()["data"]
    solution = sudoku_solver(data)
    return json.dumps({"sudoku": solution})

@app.route("/solve_step", methods=["POST"])
def solve_step():
    data = request.get_json()["data"]
    solution = sudoku_solver(data, 1) #TODO: problem when hypothesis wrong
    return json.dumps({"sudoku": solution})

@app.route("/load_preset", methods=["POST"])
def load_preset():
    level = request.get_json()["preset"]
    preset = {}
    # medium:
    preset['medium'] = [[1, 8, 0, 0, 0, 4, 5, 0, 0],
             [6, 0, 0, 0, 1, 8, 4, 0, 0],
             [0, 0, 0, 6, 7, 0, 0, 0, 0],
             [0, 0, 6, 0, 4, 0, 1, 5, 0],
             [8, 0, 0, 0, 3, 0, 0, 0, 7],
             [0, 4, 5, 0, 6, 0, 8, 0, 0],
             [0, 0, 0, 0, 8, 6, 0, 0, 0],
             [0, 0, 4, 7, 2, 0, 0, 0, 8],
             [0, 0, 8, 1, 0, 0, 0, 2, 4]]

    # easy:
    preset['easy'] = [[7, 8, 0, 3, 9, 0, 6, 4, 1],
                     [0, 0, 0, 5, 1, 0, 0, 0, 2],
                     [2, 9, 1, 7, 4, 0, 5, 8, None ],
                     [0, 0, 0, 0, 8, 0, 0, 0, 6],
                     [8, 5, 0, 0,  2, 4, 0, 0, None ],
                     [0,  1, 3, 0, 0,  7, 4, 0, None ],
                     [0, 0, 0, 0,  7, 0,  8, 3, None ],
                     [3, 0, 0, 0, 0,  1, 0, 0, None ],
                     [0,  2, 8, 0, 0, 0, 0,  6, 7]]

    # hard:
    preset['hard'] = [[0, 7, 0, 0, 2, 0, 1, 0, 0],
                     [6, 0, 3, 0, 0, 0, 0, 0, 0],
                     [2, 0, 0, 3, 0, 0, 5, 0, 0],
                     [0, 0, 0, 0, 3, 0, 0, 6, 0],
                     [0, 6, 4, 7, 0, 0, 0, 8, 0],
                     [0, 5, 0, 0, 9, 0, 0, 4, 0],
                     [0, 4, 0, 0, 7, 0, 9, 0, 0],
                     [0, 2, 0, 0, 0, 8, 0, 5, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0]]

    # expert:
    preset['expert'] = [[0, 0, 0, 0, 9, 0, 2, 0, 3],
                         [0, 0, 0, 0, 3, 0, 0, 0, 8],
                         [0, 0, 0, 5, 7, 4, 0, 0, 0],
                         [0, 0, 3, 6, 0, 0, 0, 0, 0],
                         [0, 9, 0, 0, 0, 5, 0, 0, 0],
                         [0, 2, 0, 0, 0, 0, 0, 6, 1],
                         [7, 0, 4, 0, 0, 0, 0, 3, 0],
                         [5, 0, 0, 9, 0, 0, 7, 0, 0],
                         [0, 0, 0, 0, 0, 0, 4, 0, 0]]

    return json.dumps({"sudoku": preset[level]})

def plot_soduko(soduko, title=''):
    plt.table(soduko, cellLoc='center', rowLoc='center', bbox=[0.05, 0.05, 0.9, 0.9]);
    plt.xticks([-1]);
    plt.yticks([-1]);
    plt.hlines([0, 1 / 3, 2 / 3, 1], 0, 1, linewidth=3);
    plt.vlines([0, 1 / 3, 2 / 3, 1], 0, 1, linewidth=3);
    plt.title(title)
    plt.show()

def get_cells(hypothesis):
    rows = np.array(hypothesis, dtype=object)
    cols = np.array(hypothesis, dtype=object).T
    cells = []
    for i in range(3):
        for j in range(3):
            # print('from ({},{}) to ({},{})'.format(i*3, j*3, (i+1)*3, (j+1)*3))
            cell = rows[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3]
            # print(cell)
            cells.append(cell)

    return rows,cols,cells

def do_iteration(hypothesis, opts, guess):
    changed=False
    valid=True

    for r in range(9):
        for c in range(9):
            if hypothesis[r][c] is None:
                rows, cols, cells = get_cells(hypothesis)
                cell = cells[(r // 3)*3 + c // 3]
                row = rows[r]
                col = cols[c]
                numbers_in_row_col_cell = set(row).union(set(col)).union(set(cell.flatten()))
                if len(numbers_in_row_col_cell)==10:
                    valid=False
                cur_opts = set(opts[r][c]) - set(numbers_in_row_col_cell)
                opts[r][c] = []
                rows_opts, cols_opts, cells_opts = get_cells(opts)
                for i in cur_opts:
                    row_opts = set(np.concatenate(rows_opts[r], dtype=object))
                    col_opts = set(np.concatenate(cols_opts[c], dtype=object))
                    cell_opts = set(np.concatenate(cells_opts[(r // 3)*3 + c // 3].flatten(), dtype=object))
                    if i not in row_opts or i not in col_opts or i not in cell_opts:
                        hypothesis[r][c] = i
                        print('row {} col {} set to {}'.format(r + 1, c + 1, i))
                        changed = True
                        break

                if hypothesis[r][c] is None:
                    for i in range (1,10):
                        if i not in numbers_in_row_col_cell:
                            opts[r][c].append(i)
                    if len(opts[r][c])==1:
                        hypothesis[r][c] = opts[r][c][0]
                        print('row {} col {} set to {}'.format(r+1, c+1, opts[r][c][0]))
                        changed = True
                    else:
                        if guess and len(opts[r][c])==2:
                            hypothesis1 = deepcopy(hypothesis)
                            hypothesis2 = deepcopy(hypothesis)
                            hypothesis1[r][c] = opts[r][c][0]
                            hypothesis2[r][c] = opts[r][c][1]
                            print('hypothesis1: row {} col {} set to {}'.format(r + 1, c + 1, opts[r][c][0]))
                            print('hypothesis2: row {} col {} set to {}'.format(r + 1, c + 1, opts[r][c][1]))
                            changed = True
                            return [hypothesis1, hypothesis2], opts, changed, valid

    return hypothesis, opts, changed, valid

def sudoku_solver(input, max_it_steps=1000, plot=False):
    multi = []
    if plot:
        plot_soduko(input, 'input')
    hypothesis = deepcopy(input)
    it_ctr = 0
    changed = True
    guess = False
    opts = [[[] for i in range(9)] for j in range(9)]
    while None in np.array(hypothesis).flatten() and (changed or it_ctr<2 or guess):
        it_ctr+=1
        print('************************ iteration #', it_ctr)
        hypothesis, opts, changed, valid = do_iteration(hypothesis, opts, guess)
        if not valid:
            print('current hypothesis wrong')
            hypothesis, opts = multi.pop(-1)
        if guess and changed:
            multi.append((hypothesis[1], deepcopy(opts)))
            hypothesis = hypothesis[0]

        if plot:
            plot_soduko(hypothesis, it_ctr)
        if changed or it_ctr<2:
            guess = False
        else:
            guess = True

        if valid and changed and it_ctr>=max_it_steps:
            return hypothesis

    return hypothesis

@app.route('/')
def root():
    init_detector()
    with open('index.html','r') as file:
        ret = file.read()
    return ret

def init_detector():
    global model
    model = MultiClassifier('detector/digits_classifier.pth')

def detect_img(image):
    global model
    # preproccess:
    image = image.convert("L").resize((300, 300))
    inputs = ToTensor()(image).unsqueeze(0)
    inputs = 1 - inputs # we want the background black
    outputs = model(inputs).to(int)[0,0]
    return outputs.tolist()


@app.route('/detect_sudoku', methods = ['GET', 'POST'])
def detect_sudoku():
    if request.method == 'POST':
        response = request.get_json()['file']
        response = response[response.find('base64,') + len('base64,'):]
        base64_decoded = base64.b64decode(response)
        image = Image.open(io.BytesIO(base64_decoded))
        output = detect_img(image)

        return json.dumps({"sudoku": output})


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # solve(easy)
    app.run(host='0.0.0.0', port='5000', debug=True)