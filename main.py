import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
from copy import deepcopy
import json
from flask import Flask, request, render_template
from flask_cors import CORS

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

@app.route("/load_preset", methods=["POST", "GET"])
def load_preset():
    level = request.get_json()["preset"]
    preset = {}
    # medium:
    preset['medium'] = [[1, 8, None, None, None, 4, 5, None, None],
             [6, None, None, None, 1, 8, 4, None, None],
             [None, None, None, 6, 7, None, None, None, None],
             [None, None, 6, None, 4, None, 1, 5, None],
             [8, None, None, None, 3, None, None, None, 7],
             [None, 4, 5, None, 6, None, 8, None, None],
             [None, None, None, None, 8, 6, None, None, None],
             [None, None, 4, 7, 2, None, None, None, 8],
             [None, None, 8, 1, None, None, None, 2, 4]]

    # easy:
    preset['easy'] = [[7, 8, None, 3, 9, None, 6, 4, 1],
                     [None, None, None, 5, 1, None, None, None, 2],
                     [2, 9, 1, 7, 4, None, 5, 8, None ],
                     [None, None, None, None, 8, None, None, None, 6],
                     [8, 5, None, None,  2, 4, None, None, None ],
                     [None,  1, 3, None, None,  7, 4, None, None ],
                     [None, None, None, None,  7, None,  8, 3, None ],
                     [3, None, None, None, None,  1, None, None, None ],
                     [None,  2, 8, None, None, None, None,  6, 7]]

    # hard:
    preset['hard'] = [[None, 7, None, None, 2, None, 1, None, None],
                     [6, None, 3, None, None, None, None, None, None],
                     [2, None, None, 3, None, None, 5, None, None],
                     [None, None, None, None, 3, None, None, 6, None],
                     [None, 6, 4, 7, None, None, None, 8, None],
                     [None, 5, None, None, 9, None, None, 4, None],
                     [None, 4, None, None, 7, None, 9, None, None],
                     [None, 2, None, None, None, 8, None, 5, None],
                     [None, None, None, None, None, None, None, None, None]]

    # expert:
    preset['expert'] = [[None, None, None, None, 9, None, 2, None, 3],
                         [None, None, None, None, 3, None, None, None, 8],
                         [None, None, None, 5, 7, 4, None, None, None],
                         [None, None, 3, 6, None, None, None, None, None],
                         [None, 9, None, None, None, 5, None, None, None],
                         [None, 2, None, None, None, None, None, 6, 1],
                         [7, None, 4, None, None, None, None, 3, None],
                         [5, None, None, 9, None, None, 7, None, None],
                         [None, None, None, None, None, None, 4, None, None]]

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
    # base_sudoku = [[None]*9]*9
    # sudoku1 = base_sudoku
    # sudoku1[0][0]=2
    # return render_template('sudoku.html') #, sudoku=json.dumps({"sudoku2": sudoku1}))
    with open('index.html', 'r') as file:
        ret = file.read()
    return ret

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # solve(easy)
    app.run(host='localhost', port='5000', debug=True)