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

imgElement.onload = function() {
    let image = cv.imread(imgElement);
    cv.imshow('imageCanvas', image);
    image.delete();
    points = [];
};

const canvas = document.getElementById("imageCanvas");
const ctx = canvas.getContext("2d");

// Sample 4 points from the user

canvas.addEventListener("mousedown", (event) => {
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    points.push([x, y]);
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, 2 * Math.PI);
    ctx.fillStyle = "red";
    ctx.fill();
    if (points.length == 4) {
        canvas.removeEventListener("mousedown", transformImage());
    }
});

function clear_pts() {
    console.log(points)
    points = [];
    let image = cv.imread(imgElement);
    cv.imshow('imageCanvas', image);
    image.delete();
}

function transformImage() {
    // from: https://docs.opencv.org/3.4/dd/d52/tutorial_js_geometric_transformations.html
    if (points.length !== 4) {
        return;
    }
    const [tl, tr, br, bl] = points;
    const width = Math.min(
        Math.hypot(tr[0] - tl[0], tr[1] - tl[1]),
        Math.hypot(br[0] - bl[0], br[1] - bl[1]));
    const height = Math.min(
        Math.hypot(br[0] - tr[0], br[1] - tr[1]),
        Math.hypot(bl[0] - tl[0], bl[1] - tl[1]));
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
