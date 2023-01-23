import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
from copy import deepcopy
import json
from flask import Flask, request, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/check", methods=["POST"])
def check():
    sudoku = request.get_json()["data"]
    if isvalid(sudoku):
        if completed(sudoku):
            out_msg = "Great!"
        else:
            out_msg = "So far everything good!"
    else:
        out_msg = "Something isn't right..."

    return json.dumps({"sudoku": sudoku, "success_flag": len(out_msg)==0, "out_msg": out_msg})

@app.route("/solve_all", methods=["POST"])
def solve_all():
    sudoku = request.get_json()["data"]
    if isvalid(sudoku):
        (sudoku, success_flag, out_msg) = sudoku_solver(sudoku)
    else:
        success_flag = False
        out_msg = "mistake in the board"
    return json.dumps({"sudoku": sudoku, "success_flag": success_flag, "out_msg": out_msg})

@app.route("/solve_step", methods=["POST"])
def solve_step():
    sudoku = request.get_json()["data"]
    if isvalid(sudoku):
        (sudoku, success_flag, out_msg) = sudoku_solver(sudoku, 1)  #TODO: problem when hypothesis wrong
    else:
        success_flag = False
        out_msg = "mistake in the board"
    return json.dumps({"sudoku": sudoku, "success_flag": success_flag, "out_msg": out_msg})

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

    return json.dumps({"sudoku": preset[level], "success_flag": True, "out_msg": ""})

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

    for row_ind in range(9):
        for col_ind in range(9):
            if hypothesis[row_ind][col_ind] is None:
                valid, numbers_in_row_col_cell = isvalid_rc(hypothesis, row_ind, col_ind)
                cur_opts = set(opts[row_ind][col_ind]) - set(numbers_in_row_col_cell)
                opts[row_ind][col_ind] = []
                rows_opts, cols_opts, cells_opts = get_cells(opts)
                for i in cur_opts:
                    row_opts = set(np.concatenate(rows_opts[row_ind], dtype=object))
                    col_opts = set(np.concatenate(cols_opts[col_ind], dtype=object))
                    cell_opts = set(np.concatenate(cells_opts[(row_ind // 3)*3 + col_ind // 3].flatten(), dtype=object))
                    if i not in row_opts or i not in col_opts or i not in cell_opts:
                        hypothesis[row_ind][col_ind] = i
                        print('row {} col {} set to {}'.format(row_ind + 1, col_ind + 1, i))
                        changed = True
                        break

                if hypothesis[row_ind][col_ind] is None:
                    for i in range (1, 10):
                        if i not in numbers_in_row_col_cell:
                            opts[row_ind][col_ind].append(i)
                    if len(opts[row_ind][col_ind]) == 0: # if there are no options it means hypothesis invalid
                        valid = False
                        return hypothesis, opts, changed, valid
                    elif len(opts[row_ind][col_ind])==1:
                        hypothesis[row_ind][col_ind] = opts[row_ind][col_ind][0]
                        print('row {} col {} set to {}'.format(row_ind+1, col_ind+1, opts[row_ind][col_ind][0]))
                        changed = True
                    elif guess: # and len(opts[row_ind][col_ind])>=2:
                        multihypotheses = []
                        for i in range(len(opts[row_ind][col_ind])):
                            cur_hypothesis = deepcopy(hypothesis)
                            cur_hypothesis[row_ind][col_ind] = opts[row_ind][col_ind][i]
                            print('hypothesis {}: row {} col {} set to {}'.format(i + 1, row_ind + 1, col_ind + 1,
                                                                                opts[row_ind][col_ind][i]))
                            multihypotheses.append(cur_hypothesis)

                        changed = True
                        return multihypotheses, opts, changed, valid

    return hypothesis, opts, changed, valid

def isvalid(hypothesis):
    for row_ind in range(9):
        for col_ind in range(9):
            if not isvalid_rc(hypothesis, row_ind, col_ind)[0]:
                return False
    return True

def isvalid_rc(hypothesis, row_ind, col_ind):
    valid = False
    rows, cols, cells = get_cells(hypothesis)
    cell = cells[(row_ind // 3) * 3 + col_ind // 3]
    row = rows[row_ind]
    col = cols[col_ind]
    if isvalid_list(cell.flatten()) and isvalid_list(row) and isvalid_list(col):
        valid = True

    numbers_in_row_col_cell = set(row).union(set(col)).union(set(cell.flatten()))
    return valid, numbers_in_row_col_cell

def isvalid_list(arr):
    # ignore Nones:
    arr = [a for a in arr if a is not None]
    return len(arr)==len(set(arr))

def completed(hypothesis):
    if None in np.array(hypothesis).flatten():
        return False
    else:
        return True

def sudoku_solver(input, max_it_steps=100, plot=False):
    valid = False
    multi = []
    if plot:
        plot_soduko(input, 'input')
    hypothesis = deepcopy(input)
    it_ctr = 0
    changed = True
    guess = False
    guesses_ctr = 0
    opts = [[[] for i in range(9)] for j in range(9)]
    while (not completed(hypothesis)) and (changed or it_ctr<2 or guess) and (it_ctr<=max_it_steps):
        it_ctr+=1
        print('*** iteration #', it_ctr)
        hypothesis, opts, changed, valid = do_iteration(hypothesis, opts, guess)
        if not valid or (guess and not changed):
            print('current hypothesis wrong. {} left'.format(len(multi)))
            hypothesis, opts = multi.pop(-1)
        if guess and changed: # or len(hypothesis)>1
            guesses_ctr += len(hypothesis)
            for cur_hypothesis in hypothesis[1:]:
                multi.append((cur_hypothesis, deepcopy(opts)))
            hypothesis = hypothesis[0]

        if plot:
            plot_soduko(hypothesis, it_ctr)

        # if the hypothesis hasn't change we'll try to guess next step.
        # however, on the very 1st iteration the opts where empty, now they are occupied and initial hypothesis will be given in next iteration
        if changed or it_ctr<2:
            guess = False
        else:
            guess = True

        if valid and changed and it_ctr>=max_it_steps:
            success = isvalid(hypothesis)
            out_msg = "" if success else "failed to solve"
            return (hypothesis, success, out_msg)

    success = isvalid(hypothesis) and completed(hypothesis)
    out_msg = "Solved! (needed {} guesses)".format(guesses_ctr) if success else "failed to solve after {} iterations".format(it_ctr)
    return (hypothesis, success, out_msg)

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