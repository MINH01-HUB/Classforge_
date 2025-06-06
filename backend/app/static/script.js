document.getElementById("upload-form").onsubmit = async function(e) {
    e.preventDefault();
    const formData = new FormData(e.target);

    const response = await fetch("/allocate", {
        method: "POST",
        body: formData
    });

    const data = await response.json();
    const output = document.getElementById("output");
    output.innerHTML = "<h3>Class Allocations</h3>";
    output.innerHTML += "<pre>" + JSON.stringify(data, null, 2) + "</pre>";
};