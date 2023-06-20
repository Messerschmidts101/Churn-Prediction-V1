import React from 'react'
import Papa from 'papaparse'
import { useState } from 'react'
import Input from './Input'
import Header2 from './Header2'

function FileHandler() {

    const [error, setError] = useState("")
    const [data, setData] = useState([])
    const allowedExtensions = ["csv"];
    const handleUpload = (file) => {
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
    }
    const handleParse = (csv) => {
        Papa.parse(csv, {
            header: true,
            complete: (results) => {
                const parsedData = results?.data;
                console.log(parsedData);
                // Process the parsed data as needed
                const columns = Object.keys(parsedData[0]);
                setData(columns);
            },
            error: (error) => {
                console.error('CSV parsing error:', error);
                setError('Error parsing CSV');
            }
        });
    }
    const handleRead = (file) => {
 
        // If user clicks the parse button without
        // a file we show a error
        if (!file) return setError("Enter a valid file");
 
        // Initialize a reader which allows user
        // to read any file or blob.
        const reader = new FileReader();
 
        // Event listener on reader when the file
        // loads, we parse it and set the data.
        reader.onload = ({ target }) => {
            handleParse(target.result)
            handleUpload(target.result)
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
            const fileExtension = inputFile?.name.split(".").pop();
            if (!allowedExtensions.includes(fileExtension)) {
                setError("Please input a csv file");
                return;
            }
 
            // If input type is correct set the state
            handleRead(inputFile)
        }
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
                { 
                    error ? error : "" /* data.map((col,idx) => <div key={idx}>{col}</div>) */
                }
            </div>
        </>
    )
}

export default FileHandler