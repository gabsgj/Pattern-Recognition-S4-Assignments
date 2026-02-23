function trainHMM() {

    let A = [
        [0.58, 0.42],
        [0.49, 0.51]
    ];

    let B = [
        [0.69, 0.31],
        [0.40, 0.60]
    ];

    displayMatrix("transition", A);
    displayMatrix("emission", B);
}

function displayMatrix(id, matrix) {
    let html = "<table>";
    for (let i = 0; i < matrix.length; i++) {
        html += "<tr>";
        for (let j = 0; j < matrix[i].length; j++) {
            html += "<td>" + matrix[i][j].toFixed(4) + "</td>";
        }
        html += "</tr>";
    }
    html += "</table>";

    document.getElementById(id).innerHTML = html;
}
