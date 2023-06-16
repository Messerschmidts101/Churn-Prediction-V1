import React from 'react'
import Papa from 'papaparse'
import { useState, useEffect } from 'react'
import Input from './Input'
import Header2 from './Header2'

function FileHandler() {

    let [file, setFile] = useState(null)
    const [error, setError] = useState("")
    const [data, setData] = useState([])
    const allowedExtensions = ["csv"];
    const handleParse = () => {
 
        // If user clicks the parse button without
        // a file we show a error
        if (!file) return setError("Enter a valid file");
 
        // Initialize a reader which allows user
        // to read any file or blob.
        const reader = new FileReader();
 
        // Event listener on reader when the file
        // loads, we parse it and set the data.
        reader.onload = async ({ target }) => {
            const csv = Papa.parse(target.result, { header: true });
            const parsedData = csv?.data;
            const columns = Object.keys(parsedData[0]);
            setData(columns);
        };
        reader.readAsText(file);
    };
    const handleFileChange = (event) => {
        setError("")
        if (event.target.files.length) {
            const inputFile = event.target.files[0];
 
            // Check the file extensions, if it not
            // included in the allowed extensions
            // we show the error
            const fileExtension = inputFile?.type.split("/")[1];
            if (!allowedExtensions.includes(fileExtension)) {
                setError("Please input a csv file");
                return;
            }
 
            // If input type is correct set the state
            setFile(inputFile);
        }
        fetch("http://localhost:8080/predict_dataset", {
            method: "POST",
            mode: 'cors',
            headers: {
                "Content-Type": "text/csv"
            },
            body: file
        }).then(
            data => console.log(data)
        )

    };
    return (
        <>
            <Header2>
                {
                    error ? error : "Your CSV file here!"
                }
            </Header2>
            <Input id={"file"} name={"file"} theme={"primary"} className={""} onChange={handleFileChange} type="file" placeholder={"Change File"}></Input>
            <div>
                {error ? error : data.map((col,idx) => <div key={idx}>{col}</div>)}
            </div>
        </>
    )
}

export default FileHandler