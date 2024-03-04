document.body.classList.add('loading');

function onOpenCvReady() {
    document.body.classList.remove('loading');
}

var points = [];
let imgElement = document.getElementById('imageSrc');
let inputElement = document.getElementById('fileInput');
var initial_alert = false

inputElement.onchange = function () {
    imgElement.src = URL.createObjectURL(event.target.files[0]);
};

function shadeColor(color, percent) {

    var R = parseInt(color.substring(1,3),16);
    var G = parseInt(color.substring(3,5),16);
    var B = parseInt(color.substring(5,7),16);

    R = parseInt(R * (100 + percent) / 100);
    G = parseInt(G * (100 + percent) / 100);
    B = parseInt(B * (100 + percent) / 100);

    R = (R<255)?R:255;
    G = (G<255)?G:255;
    B = (B<255)?B:255;

    R = Math.round(R)
    G = Math.round(G)
    B = Math.round(B)

    var RR = ((R.toString(16).length==1)?"0"+R.toString(16):R.toString(16));
    var GG = ((G.toString(16).length==1)?"0"+G.toString(16):G.toString(16));
    var BB = ((B.toString(16).length==1)?"0"+B.toString(16):B.toString(16));

    return "#"+RR+GG+BB;
}



function adjust(color, amount) {
    return '#' + color.replace(/^#/, '').replace(/../g, color => ('0'+Math.min(255, Math.max(0, parseInt(color, 16) + amount)).toString(16)).substr(-2));
}

// Version 4.0
const pSBC=(p,c0,c1,l)=>{
    let r,g,b,P,f,t,h,i=parseInt,m=Math.round,a=typeof(c1)=="string";
    if(typeof(p)!="number"||p<-1||p>1||typeof(c0)!="string"||(c0[0]!='r'&&c0[0]!='#')||(c1&&!a))return null;
    if(!this.pSBCr)this.pSBCr=(d)=>{
        let n=d.length,x={};
        if(n>9){
            [r,g,b,a]=d=d.split(","),n=d.length;
            if(n<3||n>4)return null;
            x.r=i(r[3]=="a"?r.slice(5):r.slice(4)),x.g=i(g),x.b=i(b),x.a=a?parseFloat(a):-1
        }else{
            if(n==8||n==6||n<4)return null;
            if(n<6)d="#"+d[1]+d[1]+d[2]+d[2]+d[3]+d[3]+(n>4?d[4]+d[4]:"");
            d=i(d.slice(1),16);
            if(n==9||n==5)x.r=d>>24&255,x.g=d>>16&255,x.b=d>>8&255,x.a=m((d&255)/0.255)/1000;
            else x.r=d>>16,x.g=d>>8&255,x.b=d&255,x.a=-1
        }return x};
    h=c0.length>9,h=a?c1.length>9?true:c1=="c"?!h:false:h,f=this.pSBCr(c0),P=p<0,t=c1&&c1!="c"?this.pSBCr(c1):P?{r:0,g:0,b:0,a:-1}:{r:255,g:255,b:255,a:-1},p=P?p*-1:p,P=1-p;
    if(!f||!t)return null;
    if(l)r=m(P*f.r+p*t.r),g=m(P*f.g+p*t.g),b=m(P*f.b+p*t.b);
    else r=m((P*f.r**2+p*t.r**2)**0.5),g=m((P*f.g**2+p*t.g**2)**0.5),b=m((P*f.b**2+p*t.b**2)**0.5);
    a=f.a,t=t.a,f=a>=0||t>=0,a=f?a<0?t:t<0?a:a*P+t*p:0;
    if(h)return"rgb"+(f?"a(":"(")+r+","+g+","+b+(f?","+m(a*1000)/1000:"")+")";
    else return"#"+(4294967296+r*16777216+g*65536+b*256+(f?m(a*255):0)).toString(16).slice(1,f?undefined:-2)
}

const RGB_Linear_Shade=(p,c)=>{
    var i=parseInt,r=Math.round,[a,b,c,d]=c.split(","),P=p<0,t=P?0:255*p,P=P?1+p:1-p;
    return"rgb"+(d?"a(":"(")+r(i(a[3]=="a"?a.slice(5):a.slice(4))*P+t)+","+r(i(b)*P+t)+","+r(i(c)*P+t)+(d?","+d:")");
}

// var cur_col=[]

function mouseover_cell(x,y) {

    var cell = document.getElementById("cell_" + x + "_" + y);

    var cur_cell_col = window.getComputedStyle(cell).backgroundColor;
    var cur_col = []
    for (var i = 0; i < 9; i++) {
        var cell = document.getElementById("cell_" + i + "_" + y);
        cur_col.push(window.getComputedStyle(cell).backgroundColor);
        cell.style.backgroundColor = pSBC(-0.6, cur_col[i])
    }
    for (var i = 0; i < 9; i++) {
        var cell = document.getElementById("cell_" + x + "_" + i);
        cur_col.push(window.getComputedStyle(cell).backgroundColor);
        cell.style.backgroundColor = pSBC(-0.6, cur_col[9+i])
    }

    cur_col[x]=cur_cell_col
    cur_col[9+y]=cur_cell_col
    window.cur_col = cur_col
}

function mouseleave_cell(x,y) {

    cur_col = window.cur_col
    // console.log(cur_col)
    for (var i = 0; i < 9; i++) {
        var cell = document.getElementById("cell_" + i + "_" + y);
        cell.style.backgroundColor = cur_col[i]

    }
    for (var i = 0; i < 9; i++) {
        var cell = document.getElementById("cell_" + x + "_" + i);
        cell.style.backgroundColor = cur_col[9+i]
    }
}


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
    clear_pts()
    if (initial_alert==true) {
        alert('Mark the Sudoku corners with 4 points in the following order: top-left, top-right, bottom-right, bottom-left. Then press "detect"')
        initial_alert = false
    }
};

const canvas = document.getElementById("imageCanvas");
const ctx = canvas.getContext("2d");

// Sample 4 points from the user

canvas.addEventListener("mousedown", (event) => {
    const rect = canvas.getBoundingClientRect();
    // console.log(rect)
    console.log(event.clientX, event.clientY)

    ctx.strokeStyle = "red";
    ctx.lineWidth=3;

    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    points.push([x, y]);
    console.log(points)
    // ctx.beginPath();
    // first point: (pi/2,-pi/2), second: (-pi/2, pi), third: (-pi, pi/2), fourth: (pi/2, 2*pi)
    if (points.length==1) {
        ctx.beginPath()
        ctx.arc(x, y, 10, Math.PI / 2, 0)
        ctx.stroke()

        ctx.beginPath()
        ctx.moveTo(x, y)
        ctx.stroke()

    }
    if (points.length==2) {
        ctx.lineTo(x,y)
        ctx.stroke()

        ctx.arc(x, y, 10,  Math.PI, Math.PI/2)
        ctx.stroke()

    };
    if (points.length==3) {
        ctx.lineTo(x,y)
        ctx.stroke()

        ctx.arc(x, y, 10,  3*Math.PI/2, Math.PI)
        ctx.stroke()

        };
    if (points.length==4) {
        ctx.lineTo(x,y)
        ctx.stroke()

        ctx.arc(x, y, 10,  0, 3*Math.PI/2)
        ctx.stroke()

        ctx.lineTo(points[0][0], points[0][1])
        ctx.stroke()

        };
    // if (points.length == 4) {
    //     canvas.removeEventListener("mousedown", transformImage());
    // }
});

// canvas.addEventListener("mouseover", (event) => {
//     const rect = canvas.getBoundingClientRect();
//     // console.log(rect)
//     // console.log(event.clientX, event.clientY)
//
//     const x = event.clientX - rect.left;
//     const y = event.clientY - rect.top;
//     // points.push([x, y]);
//     // ctx.beginPath();
//     // first point: (pi/2,-pi/2), second: (-pi/2, pi), third: (-pi, pi/2), fourth: (pi/2, 2*pi)
//     // if (points.length==0) {
//     //     ctx.beginPath()
//     //     ctx.arc(x, y, 10, Math.PI / 2, 0)
//         // ctx.beginPath()
//         // ctx.moveTo(x, y)
//     // }
//     if (points.length>1) {
//         ctx.lineTo(x,y)
//         // ctx.arc(x, y, 10,  Math.PI, Math.PI/2)
//     };
//     // if (points.length==3) {
//     //     ctx.lineTo(x,y)
//     //     // ctx.arc(x, y, 10,  3*Math.PI/2, Math.PI)
//     //     };
//     // if (points.length==4) {
//     //     ctx.lineTo(x,y)
//     //     // ctx.arc(x, y, 10,  0, 3*Math.PI/2)
//     //     };
//     ctx.strokeStyle = "red";
//     ctx.lineWidth=5;
//     // ctx.fill();
//     ctx.stroke()
//
// });

function clear_pts() {
    console.log(points)
    points = [];
    resize_show_img()
}

function transformImage() {
    // from: https://docs.opencv.org/3.4/dd/d52/tutorial_js_geometric_transformations.html
    if (points.length !== 4) {
        return;
    }
    const [tl, tr, br, bl] = points;
    const width = 600 //Math.min(
    // Math.hypot(tr[0] - tl[0], tr[1] - tl[1]),
    // Math.hypot(br[0] - bl[0], br[1] - bl[1]));
    const height = 600 //Math.min(
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
                        cell.style.backgroundColor = "#fdd9d9"
                    }
                } else {
                    cell.value = data[i][j];
                    if (init == true) {
                        cell.style.backgroundColor = "#fdd9d9"
                    } else {
                        cell.style.backgroundColor = "#a1cece"
                    }
                }
            } else {
                cell.style.backgroundColor = "#ffffff"
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
            show_sudoku()
            show_sudoku(response.sudoku)
        }
    };
};

