function store_sudoku() {
    var data = [];
    for (var i = 0; i < 9; i++) {
        var row = [];
        for (var j = 0; j < 9; j++) {
            var cell = document.getElementById("cell_" + i + "_" + j);
            if (cell.value.length == 0) {
                row.push(null);
            } else {
                row.push(parseInt(cell.value));
            }
        }
        data.push(row);
    }
    return data
};

function show_sudoku(data = Array(9).fill(null).map(() => Array(9).fill(null))) {
    for (var i = 0; i < 9; i++) {
        for (var j = 0; j < 9; j++) {
            var cell = document.getElementById("cell_" + i + "_" + j);
            cell.value = data[i][j];
        }
    }
    ;
};

function solve_all() {
    var data = store_sudoku();
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "http://localhost:5000/solve", true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
            var response = JSON.parse(xhr.responseText);
            console.log(response);
            show_sudoku(response.sudoku)
        }
    };
    xhr.send(JSON.stringify({data: data}));
};

function solve_step() {
    var data = store_sudoku();
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "http://localhost:5000/solve_step", true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
            var response = JSON.parse(xhr.responseText);
            console.log(response);
            show_sudoku(response.sudoku)
        }
    };
    xhr.send(JSON.stringify({data: data}));
};

function load_preset() {
    var preset_selection = document.getElementById("preset-select").value;
    // Send the chosen preset to the Python function "load_preset"
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "http://localhost:5000/load_preset", true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.send(JSON.stringify({preset: preset_selection}));
    xhr.onreadystatechange = function () {
    if (xhr.readyState === 4 && xhr.status === 200) {
        var response = JSON.parse(xhr.responseText);
        console.log(response);
        show_sudoku(response.sudoku)
    }
    };
};

function load_preset2() {
    var preset_selection = document.getElementById("preset-select").value;
    // Send the chosen preset to the Python function "load_preset"
    fetch("http://localhost:5000/load_preset",
        {method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({preset: preset_selection})})
    .then(response => response.json())
    // .then(json => console.log(json))
    .then(json => show_sudoku(json.sudoku))
    };