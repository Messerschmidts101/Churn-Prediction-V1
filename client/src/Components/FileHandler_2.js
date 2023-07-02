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
    const handleFileChange = (event) => {
        setError('');
    
        const inputFile = event.target.files[0];
    
        if (!inputFile) {
            setError('No file selected');
            return;
        }
    
        const fileExtension = inputFile.name.split('.').pop();
    
        if (!allowedExtensions.includes(fileExtension)) {
            setError('Please input a CSV file');
            return;
        }
    
        setFile(inputFile);
        const reader = new FileReader();

        reader.onload = ({ target }) => {
            const csvData = target.result;
            parseCSV(csvData);
            uploadFile(csvData);
        };

        reader.readAsText(inputFile);
        
        const uploadFile = (csvData) => {
            fetch('http://localhost:8080/predict_dataset', {
                method: 'POST',
                headers: {
                'Content-Type': 'text/csv'
                },
                body: csvData
            })
                .then((response) => response.json())
                .then((data) => {
                console.log('File uploaded:', data);
                // Handle the response from the server
                })
                .catch((error) => {
                console.error('Error uploading file:', error);
                });
        };
    
        const parseCSV = (csvData) => {
        Papa.parse(csvData, {
            header: true,
            complete: (results) => {
            const parsedData = results.data;
            console.log(parsedData);
            // Process the parsed data as needed
            },
            error: (error) => {
            console.error('CSV parsing error:', error);
            setError('Error parsing CSV');
            }
        });
        };
    }
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