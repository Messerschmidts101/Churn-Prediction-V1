import React from 'react'
import Highcharts from 'highcharts'
import HighchartsReact from 'highcharts-react-official'
import { useState, useEffect } from 'react'
import Header1 from '../Components/Header1'
import Header2 from '../Components/Header2'
import Dropdown from 'react-dropdown';
import Papa from 'papaparse'
import Input from '../Components/Input'

function MultipleCustomers() {
    
    let [series, setSeries] = useState([{}])
    const [error, setError] = useState("")
    const [feature_list, setFeatureList] = useState([{}])
    useEffect(() => {
        fetch("/feature_names").then(
            res => res.json()
        ).then(
            data => {
                setFeatureList(data)
            }
        )
    }, [])
    const [featureName, setFeatureName] = useState("total_day_calls")
    const allowedExtensions = ["csv"];
    const handleMapper = (name, data) => {
        console.log("data: ",data)
        let obj =  data[name]
        return Object.keys(obj).map((key) => obj[key])
    }
    const handleTransformation = (data) => {
        let prediction = handleMapper("prediction", data)
        let feature =  handleMapper(featureName, data)
        const totalSum = feature.reduce((sum, value) => sum + value, 0)
        const ratio = Math.max.apply(Math, feature) / 100
        const percentageData = feature.map(value => (value / totalSum) * 100);
        feature = percentageData.map(value => (value / ratio) > 1 ? 1 : value / ratio);
        setSeries(
            [
                {
                    'name': featureName,
                    'data': feature
                },
                {
                    'name': 'prediction',
                    'data': prediction
                },
            ]
        )
    }
    const handleUpload = (file) => {
        fetch("http://localhost:8080/predict_dataset_feature", {
            method: "POST",
            mode: 'cors',
            headers: {
                "Content-Type": "text/csv"
            },
            body: file
        }).then(
            res => res.json()
        ).then(
            data => {
                handleTransformation(data)
                console.log("data: ",data[0])
                console.log("data: ",data)
            }
        )
    }
    const handleParse = (csv) => {
        Papa.parse(csv, {
            header: true,
            complete: (results) => {
                const parsedData = results?.data;
                console.log(parsedData);
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
    const handleDropdown = (feature) => {
        setFeatureName(feature)
    }

    
    var TITLE = 'Churn of Customers';
    var CAPTION = 'Source: Telco Company';
    CAPTION += '<div style="display:block;">TODO</div>';
    var X_AXIS = 'Customers';
    var Y_AXIS = 'Churn Rate';
    var BEGIN_AT_ZERO = true;
    var SHOW_LEGEND = true;

    const optionsLine = {

        series: series,

        chart: {
            type: 'area',
            zoomType: 'x',
            panning: true,
            panKey: 'shift',
            scrollablePlotArea: {
                minWidth: 600
            }
        },

        title: { text: TITLE },
        caption: { text: CAPTION },
        credits: { enabled: false },

        annotations: [{
            labelOptions: {
                backgroundColor: 'rgba(255,255,255,0.8)',
                verticalAlign: 'top',
                y: 10
            },
            labels: "annotationPoints"
        }],

        xAxis: {
            title: { text: X_AXIS },
        },

        yAxis: {
            startOnTick: true,
            min: BEGIN_AT_ZERO ? 0 : null,
            title: { text: Y_AXIS },
            labels: { formatter: (x) => {return x.value.toLocaleString()} }
        },

        legend: { enabled: SHOW_LEGEND },
    }

    const optionsPie = {
        series: series,

        chart: {
            type: 'pie',
            zoomType: 'x',
            panning: true,
            panKey: 'shift',
            scrollablePlotArea: {
                minWidth: 600
            }
        },

        title: { text: "Customer Attrition Percentage" },
        caption: { text: CAPTION },
        credits: { enabled: false },

        annotations: [{
            labelOptions: {
                backgroundColor: 'rgba(255,255,255,0.8)',
                verticalAlign: 'top',
                y: 10
            },
            labels: "annotationPoints"
        }],

        xAxis: {
            title: { text: X_AXIS },
        },

        yAxis: {
            startOnTick: true,
            min: BEGIN_AT_ZERO ? 0 : null,
            title: { text: Y_AXIS },
            labels: { formatter: (x) => {return x.value.toLocaleString()} }
        },

        legend: { enabled: SHOW_LEGEND },
    }

    return (
        <>
            <div className='container'>
                <Header1 className={"text-center lable1 pt-5"}>Predict a dataset of customer's churn!</Header1>
                <Header2>
                    {
                        error ? error : "Your CSV file here!"
                    }
                </Header2>
                <Input id={"file"} name={"file"} theme={"primary"} className={""} onChange={handleFileChange} type="file" placeholder={"Change File"} />
                {/* <Dropdown options={feature_list} onChange={handleDropdown} value={"prediction"} placeholder="Select an option" />; */}
                <div>
                    { 
                        error ? error : "" /* data.map((col,idx) => <div key={idx}>{col}</div>) */
                    }
                </div>
            </div>
            <div className='container'>
                <HighchartsReact highcharts={Highcharts} options={optionsLine} />
                <HighchartsReact highcharts={Highcharts} options={optionsPie} />
            </div>
        </>
    )
}

export default MultipleCustomers