document.body.classList.add('loading');

function onOpenCvReady() {
    document.body.classList.remove('loading');
}

var points = [];
let imgElement = document.getElementById('imageSrc');
let inputElement = document.getElementById('fileInput');

inputElement.onchange = function () {
    imgElement.src = URL.createObjectURL(event.target.files[0]);
};

function resize_show_img(){
    let image = cv.imread(imgElement);
    console.log('original image size: ', image.rows, image.cols)
    console.log('page width: ', document.documentElement.clientWidth)
    let dst = new cv.Mat();
    let new_width = Math.round(document.documentElement.clientWidth*0.95)
    let new_height = Math.round(new_width/image.cols*image.rows)
    let dsize = new cv.Size(new_width, new_height);
    cv.resize(image, dst, dsize, 0, 0, cv.INTER_AREA);
    console.log('resized to: ', dst.rows, dst.cols)
    cv.imshow('imageCanvas', dst);
    image.delete();
}

imgElement.onload = function () {
    resize_show_img()
    points = [];
    alert('Mark the Sudoku corners with 4 points in the following order: top-left, top-right, bottom-right, bottom-left. Then press "detect"')
};

const canvas = document.getElementById("imageCanvas");
const ctx = canvas.getContext("2d");

// Sample 4 points from the user

canvas.addEventListener("mousedown", (event) => {
    const rect = canvas.getBoundingClientRect();
    console.log(rect)
    console.log(event.clientX, event.clientY)
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    points.push([x, y]);
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, 2 * Math.PI);
    ctx.fillStyle = "red";
    ctx.fill();
    if (points.length == 4) {
        canvas.removeEventListener("mousedown", transformImage());
    }
});

function clear_pts() {
    console.log(points)
    points = [];
    // let image = cv.imread(imgElement);
    // cv.imshow('imageCanvas', image);
    // image.delete();
    resize_show_img()
}

function transformImage() {
    // from: https://docs.opencv.org/3.4/dd/d52/tutorial_js_geometric_transformations.html
    if (points.length !== 4) {
        return;
    }
    const [tl, tr, br, bl] = points;
    const width = 300 //Math.min(
    // Math.hypot(tr[0] - tl[0], tr[1] - tl[1]),
    // Math.hypot(br[0] - bl[0], br[1] - bl[1]));
    const height = 300 //Math.min(
    // Math.hypot(br[0] - tr[0], br[1] - tr[1]),
    // Math.hypot(bl[0] - tl[0], bl[1] - tl[1]));
    const srcCorners = [tl[0], tl[1], tr[0], tr[1], br[0], br[1], bl[0], bl[1]];
    const dstCorners = [0, 0, width - 1, 0, width - 1, height - 1, 0, height - 1];
    let srcTri = cv.matFromArray(4, 1, cv.CV_32FC2, srcCorners);
    let dstTri = cv.matFromArray(4, 1, cv.CV_32FC2, dstCorners);
    let M = cv.getPerspectiveTransform(srcTri, dstTri);

    let src = cv.imread('imageCanvas');
    let dst = new cv.Mat();
    // let dsize = new cv.Size(src.rows, src.cols);
    let dsize = new cv.Size(width, height);
    cv.warpPerspective(src, dst, M, dsize, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());
    cv.imshow('imageCanvas', dst);
    src.delete();
    dst.delete();
    M.delete();
    srcTri.delete();
    dstTri.delete();
}

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

function show_sudoku(data = Array(9).fill(0).map(() => Array(9).fill(0)), init = false) {
    console.log('received array: ')
    console.log(data)
    for (var i = 0; i < 9; i++) {
        for (var j = 0; j < 9; j++) {
            var cell = document.getElementById("cell_" + i + "_" + j);
            if (data[i][j]) {
                if (cell.value.length == 1) {
                    if (init == false) {
                        cell.style.backgroundColor = "gray"
                    }
                } else {
                    cell.value = data[i][j];
                    if (init == true) {
                        cell.style.backgroundColor = "gray"
                    } else {
                        cell.style.backgroundColor = "lightblue"
                    }
                }
            } else {
                cell.style.backgroundColor = "white"
                cell.value = ""
            }
        }
    }
}

function solve_all() {
    var data = store_sudoku();
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/solve", true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
            var response = JSON.parse(xhr.responseText);
            console.log(response);
            if (response.success_flag == true) {
                show_sudoku(response.sudoku)
            } else {
                alert(response.out_msg)
            }
        }
    };
    xhr.send(JSON.stringify({data: data}));
};

function process_request(func_name) {
    var data = store_sudoku();
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/" + func_name, true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
            var response = JSON.parse(xhr.responseText);
            console.log(response);
            show_sudoku(response.sudoku);
            if (response.out_msg.length > 0) {
                alert(response.out_msg)
            }
        }
    }
    xhr.send(JSON.stringify({data: data}));
};

// function get_preset(level) {
//     const preset = {};
// //   medium:
//         preset.medium = [[1, 8, 0, 0, 0, 4, 5, 0, 0],
//             [6, 0, 0, 0, 1, 8, 4, 0, 0],
//             [0, 0, 0, 6, 7, 0, 0, 0, 0],
//             [0, 0, 6, 0, 4, 0, 1, 5, 0],
//             [8, 0, 0, 0, 3, 0, 0, 0, 7],
//             [0, 4, 5, 0, 6, 0, 8, 0, 0],
//             [0, 0, 0, 0, 8, 6, 0, 0, 0],
//             [0, 0, 4, 7, 2, 0, 0, 0, 8],
//             [0, 0, 8, 1, 0, 0, 0, 2, 4]]
//
//
// //    easy:
//         preset.easy = [[7, 8, 0, 3, 9, 0, 6, 4, 1],
//             [0, 0, 0, 5, 1, 0, 0, 0, 2],
//             [2, 9, 1, 7, 4, 0, 5, 8, 0],
//             [0, 0, 0, 0, 8, 0, 0, 0, 6],
//             [8, 5, 0, 0, 2, 4, 0, 0, 0],
//             [0, 1, 3, 0, 0, 7, 4, 0, 0],
//             [0, 0, 0, 0, 7, 0, 8, 3, 0],
//             [3, 0, 0, 0, 0, 1, 0, 0, 0],
//             [0, 2, 8, 0, 0, 0, 0, 6, 7]]
//
//
// //    hard:
//         preset.hard = [[0, 7, 0, 0, 2, 0, 1, 0, 0],
//             [6, 0, 3, 0, 0, 0, 0, 0, 0],
//             [2, 0, 0, 3, 0, 0, 5, 0, 0],
//             [0, 0, 0, 0, 3, 0, 0, 6, 0],
//             [0, 6, 4, 7, 0, 0, 0, 8, 0],
//             [0, 5, 0, 0, 9, 0, 0, 4, 0],
//             [0, 4, 0, 0, 7, 0, 9, 0, 0],
//             [0, 2, 0, 0, 0, 8, 0, 5, 0],
//             [0, 0, 0, 0, 0, 0, 0, 0, 0]]
//
//
// //    expert:
//         preset.expert = [[0, 0, 0, 0, 9, 0, 2, 0, 3],
//             [0, 0, 0, 0, 3, 0, 0, 0, 8],
//             [0, 0, 0, 5, 7, 4, 0, 0, 0],
//             [0, 0, 3, 6, 0, 0, 0, 0, 0],
//             [0, 9, 0, 0, 0, 5, 0, 0, 0],
//             [0, 2, 0, 0, 0, 0, 0, 6, 1],
//             [7, 0, 4, 0, 0, 0, 0, 3, 0],
//             [5, 0, 0, 9, 0, 0, 7, 0, 0],
//             [0, 0, 0, 0, 0, 0, 4, 0, 0]]
//
//     return JSON.stringify({"sudoku": preset[level], "success_flag": true, "out_msg": ""})
// };


function load_preset() {
    var preset_selection = document.getElementById("preset-select").value;
    // Send the chosen preset to the Python function "load_preset"
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/load_preset", true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.send(JSON.stringify({preset: preset_selection}));
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
            var response = JSON.parse(xhr.responseText);
    // var response = get_preset(preset_selection)
            console.log(response);
            show_sudoku();
            show_sudoku(response.sudoku, true);
}}};

// function load_preset2() {
//     var preset_selection = document.getElementById("preset-select").value;
//     // Send the chosen preset to the Python function "load_preset"
//     fetch("/load_preset",
//         {
//             method: 'POST',
//             headers: {'Content-Type': 'application/json'},
//             body: JSON.stringify({preset: preset_selection})
//         })
//         .then(response => response.json())
//         // .then(json => console.log(json))
//         .then(json => show_sudoku(json.sudoku))
// };

function detect_sudoku() {
    var dataURL = canvas.toDataURL("image/png")
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/detect_sudoku", true);
    xhr.setRequestHeader("Content-Type", "application/json");
    // console.log(JSON.stringify({file: dataURL}))
    xhr.send(JSON.stringify({file: dataURL}));
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
            var response = JSON.parse(xhr.responseText);
            console.log(response);
            show_sudoku(response.sudoku)
        }
    };
};

