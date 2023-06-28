import React from 'react'
import Highcharts from 'highcharts'
import HighchartsReact from 'highcharts-react-official'
import { useState, useEffect } from 'react'
import Header1 from '../Components/Header1'
import Header2 from '../Components/Header2'
import Papa from 'papaparse'
import Input from '../Components/Input'

function MultipleCustomers() {
    
    const [seriesArea, setSeriesArea] = useState([{}])
    const [seriesBar, setSeriesBar] = useState([{}])
    const [seriesPie, setSeriesPie] = useState([{}])
    const [population, setPopulation] = useState()
    const [data, setData] = useState([{}])
    const [error, setError] = useState("")
    const [feature_list, setFeatureList] = useState([{}])
    const [selector_area_1, setSelectorArea1] = useState('churn')
    const [selector_area_2, setSelectorArea2] = useState('state')
    const [selector_bar_1, setSelectorBar1] = useState('state')
    const [selector_bar_2, setSelectorBar2] = useState('state')
    useEffect(() => {
        fetch("/feature_names").then(
            res => res.json()
        ).then(
            data => {
                setFeatureList(data)
            }
        )
    }, [])
    const allowedExtensions = ["csv"];
    const handleMapper = (name, data) => {
        console.log("data: ",data)
        let obj =  data[name]
        return Object.keys(obj).map((key) => obj[key])
    }
    const handleAreaTransformation = (data) => {
        console.log("selector_area_1:", selector_area_1)
        console.log("selector_area_2:", selector_area_2)
        let prediction = handleMapper(selector_area_1, data)
        let feature =  handleMapper(selector_area_2, data)
        const totalSum = feature.reduce((sum, value) => sum + value, 0)
        const ratio = Math.max.apply(Math, feature) / 100
        const percentageData = feature.map(value => (value / totalSum) * 100);
        feature = percentageData.map(value => (value / ratio) > 1 ? 1 : value / ratio);
        setSeriesArea(
            [
                {
                    'name': selector_area_1,
                    'data': feature
                },
                {
                    'name': selector_area_2,
                    'data': prediction
                },
            ]
        )
        handleBarTransformation(data)
    }
    const handleBarTransformation = (data) => {
        console.log("selector_bar_1: ", selector_bar_1)
        console.log("selector_bar_2: ", selector_bar_2)
        let bar1 = handleMapper(selector_bar_1, data)
        let bar2 = handleMapper(selector_bar_2, data)
        bar1 = bar1.reduce((accumulator, accumulated) => { return accumulator + accumulated }, 0)
        bar2 = bar2.reduce((accumulator, accumulated) => { return accumulator + accumulated }, 0)
        setSeriesBar(
            [
                {
                    'name': selector_bar_1,
                    'data': [bar1]
                },
                {
                    'name': selector_bar_2,
                    'data': [bar2]
                },
            ]
        )
        console.log("bar1", bar1)
        console.log("bar2", bar2)
    }
    const handlePie = (data) => {
        let prediction = handleMapper("churn", data)
        let range = 10
        let result = prediction.reduce((result, value) => {
            var slot = Math.floor(((value - 0.01) / range) * 100);
            (result[slot] = result[slot] || []).push(value);
            return result;
        }, [])
        console.log("pie: ", result)
        let datum = []
        result.forEach(value => {
            console.log("value:", value)
            let min = Math.ceil(Math.min.apply(Math, value) * 100)
            let max = Math.ceil(Math.max.apply(Math, value) * 100)
            datum.push({
                "name": "Range " + min + "%-" + max + "%",
                "y": value.length
            })
            setPopulation(datum)
        })//[result[0].length, result[1].length, result[2].length, result[3].length, result[4].length, result[5].length, result[6].length, result[7].length, result[9].length, result[-1].length]
        console.log("datum", datum)
        setSeriesPie([
            {
                'name': 'churn',
                'data': datum
            },
        ])

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
                setData(data)
                handleAreaTransformation(data)
                handlePie(data)
                handleBarTransformation(data)
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

    var CAPTION = 'Source: Telco Company';
    CAPTION += '<div style="display:block;">TODO</div>';
    var X_AXIS = 'Customers';
    var Y_AXIS = 'Churn Rate';
    var BEGIN_AT_ZERO = true;
    var SHOW_LEGEND = true;
    const colours = ["#14342bff", "#60935dff", "#bab700ff", "#bbdfc5ff", "#ff579fff", "#faa33fff"]
    // --dark-green: #14342bff;
    // --asparagus: #60935dff;
    // --old-gold: #bab700ff;
    // --celadon: #bbdfc5ff;
    // --brilliant-rose: #ff579fff;
    // --orange: #faa33fff;

    const optionsArea = {

        series: seriesArea,

        chart: {
            type: 'area',
            zoomType: 'x',
            panning: true,
            panKey: 'shift',
            scrollablePlotArea: {
                minWidth: 600
            }
        },

        title: { text: 'Churn of Customers' },
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
        colors: colours,

        legend: { enabled: SHOW_LEGEND },
    }
    const optionsPie = {
        series: seriesPie,

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
        colors: colours,

        legend: { enabled: SHOW_LEGEND },
    }
    const optionsBar = {
        series: seriesBar,

        chart: {
            type: 'bar',
            zoomType: 'x',
            panning: true,
            panKey: 'shift',
            scrollablePlotArea: {
                minWidth: 600
            }
        },

        title: { text: "Comparison of Features" },
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
            title: { text: "Feature" },
        },

        yAxis: {
            startOnTick: true,
            min: BEGIN_AT_ZERO ? 0 : null,
            title: { text: "Sum" },
            labels: { formatter: (x) => {return x.value.toLocaleString()} }
        },
        colors: colours,

        legend: { enabled: SHOW_LEGEND },
    }

    return (
        <>
            <main>
                <div className='container'>
                    <div className='row'>
                        <Header1 className={"text-center lable1 pt-5"}>Predict a dataset of customer's churn!</Header1>
                        <Header2>Your CSV file here!</Header2>
                    </div>
                    <div role='alert'>
                        {   error ? (<div className='alert alert-danger'><span className=''>{error}</span></div>) : ""  }
                    </div>
                    <div className='row'>    
                        <Input id={"file"} name={"file"} theme={"primary"} className={"input form-control"} onChange={handleFileChange} type="file" placeholder={"Change File"} />
                    </div>
                </div>
                <div className='container'>
                    <div className='row'>
                        <div className='col mt-5'>
                            <div className='d-flex flex-row my-3'>
                                {
                                    population ? population.map((range) => {
                                        return ( <>
                                            <div key={range.name} className='col-1'>{range.name}<br /><b>{range.y}</b></div>
                                        </>)
                                    }) : "Nothing here"
                                }
                            </div>
                            <HighchartsReact highcharts={Highcharts} options={optionsPie} />
                        </div>
                        <div className='col mt-5'>
                            <div  className='d-flex flex-row my-3'>
                                <div className='col-1'>
                                    <p className=' align-middle h-100' style={{border:"0", margin:"0"}}>Features:</p>
                                </div>
                                <div className='col-3'>
                                    <select id='bar-selector' className='form-select' onChange={(event) => { setSelectorBar1(event.target.value); handleBarTransformation(data) }}>
                                        {   feature_list.map((feature) => { return ( <option key={feature.name + "bar1"} value={feature.name}>{feature.name}</option> ) })  }
                                    </select>
                                </div>
                                <div className='col-3'>
                                    <select id='bar-selector' className='form-select' onChange={(event) => { setSelectorBar2(event.target.value); handleBarTransformation(data) }}>
                                        {   feature_list.map((feature) => { return ( <option key={feature.name + "bar1"} value={feature.name}>{feature.name}</option> ) })  }
                                    </select>
                                </div>
                            </div>
                            <HighchartsReact highcharts={Highcharts} options={optionsBar} />  
                        </div>
                    </div>
                    <div className='row'>
                        <div className='d-flex flex-row mb-3 mt-5'>
                            <div className='col-1'>
                                <p className=' align-middle h-100' style={{border:"0", margin:"0"}}>Features:</p>
                            </div>
                            <div className='col-3'>
                                <select id='area-selector' className='form-select' onChange={(event) => { setSelectorArea1(event.target.value); handleAreaTransformation(data) } }>
                                    {   feature_list.map((feature) => { return ( <option value={feature.name} selected={"churn" === feature.name ? true : false} >{feature.name}</option> ) })  }
                                </select>
                            </div>
                            <div className='col-3'>
                                <select id='area-selector' className='form-select' onChange={(event) => { setSelectorArea2(event.target.value); handleAreaTransformation(data) } }>
                                    {   feature_list.map((feature) => { return ( <option value={feature.name} >{feature.name}</option> ) })  }
                                </select>
                            </div>
                        </div>
                        <HighchartsReact highcharts={Highcharts} options={optionsArea} />
                    </div>
                </div>
            </main>
        </>
    )
}

export default MultipleCustomers