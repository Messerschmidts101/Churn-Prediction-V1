import React from 'react'
import Highcharts from 'highcharts'
import HighchartsReact from 'highcharts-react-official'
import { useState, useEffect } from 'react'
import Header3 from '../components/Header3'
import Papa from 'papaparse'
import Input from '../components/Input'
import Heroe from '../components/Heroe'

function MultipleCustomers() {
    
    const [seriesArea, setSeriesArea] = useState([{}])
    const [seriesBar, setSeriesBar] = useState([{}])
    const [seriesPie, setSeriesPie] = useState([{}])
    const [population, setPopulation] = useState()
    const [data, setData] = useState([{}])
    const [error, setError] = useState("")
    const [feature_list, setFeatureList] = useState([{}])
    const [selector_area_1, setSelectorArea1] = useState('churn')
    const [selector_area_2, setSelectorArea2] = useState('total_day_calls')
    const [selector_bar_1, setSelectorBar1] = useState('total_day_calls')
    const [selector_bar_2, setSelectorBar2] = useState('total_eve_calls')
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
    const handleNormalization = (array) => {
        // let ratio = Math.max.apply(Math, array) / 100
        // for(let i = 0; i < array.lengthl; i++) {
        //     array[i] /= ratio
        // }
        // return array
        var max = array.reduce(function (p, c) {
            return p < c ? c : p;
        }, 0);
        return array.map(function (d, i) {
            return (d / max);
            // could be more efficient with just
            // `(d / max) | 0`, if you divide `max` by 100 above
        });
    }
    const handleAreaTransformation = (data) => {
        console.log("selector_area_1:", selector_area_1)
        console.log("selector_area_2:", selector_area_2)
        let prediction = handleMapper(selector_area_1, data)
        let feature =  handleMapper(selector_area_2, data)
        feature = handleNormalization(feature)
        // const totalSum = feature.reduce((sum, value) => sum + value, 0)
        // const ratio = Math.max.apply(Math, feature) / 100
        // const percentageData = feature.map(value => (value / totalSum) * 100);
        // feature = percentageData.map(value => (value / ratio) > 1 ? 1 : value / ratio);
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
        fetch("http://localhost:8080/predict_dataset", {
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

    var CAPTION = 'Source: Telco Generic';
    var X_AXIS = 'Customers';
    var Y_AXIS = 'Churn Rate';
    var BEGIN_AT_ZERO = true;
    var SHOW_LEGEND = true;
    var BACKGROUNDCOLOR = "#00000000"
    var HEIGHT = 400
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
            backgroundColor:  BACKGROUNDCOLOR,
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
            backgroundColor:  BACKGROUNDCOLOR,
            panning: true,
            panKey: 'shift',
            height: HEIGHT,
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
            backgroundColor:  BACKGROUNDCOLOR,
            panning: true,
            panKey: 'shift',
            height: HEIGHT,
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
                    <Heroe display={"display-1"} subheading={"Predict a customer dataset's churn"} boolSubheading={true}>Multiple Customer Churn</Heroe>
                    <div className='card mt-5'>
                        <div className='d-flex flex-row w-100 my-4'>
                            <div className='col-md-2'>
                                <Header3 className={"my-auto"}>Select CSV file</Header3>
                            </div>
                            <div className='col-md-9'>
                                <Input id={"file"} name={"file"} theme={"primary"} className={"input form-control"} onChange={handleFileChange} type="file" placeholder={"Change File"} />
                            </div>
                        </div>
                    </div>
                    <div role='alert'>
                        {   error ? (<div className='alert alert-danger'><span className=''>{error}</span></div>) : ""  }
                    </div>
                </div>
                <div className='container'>
                    <div className='row'>
                        <div className='col-md-6 col-sm-9 my-auto'>
                            <div  className='card graph'>
                                <div className='d-flex flex-row my-3 card-title w-100'>
                                    {
                                        population ? population.map((range) => {
                                            return ( <>
                                                <div key={range.name} className='col-1'>{range.name}<br /><b>{range.y}</b></div>
                                            </>)
                                        }) : "Nothing here"
                                    }
                                </div>
                                <div className='card-body w-100'>
                                    <HighchartsReact highcharts={Highcharts} options={optionsPie} />
                                </div>
                            </div>
                        </div>
                        <div className='col-md-6 col-sm-9 my-auto'>
                            <div className='card graph'>
                                <div  className='d-flex flex-row my-3 card-title w-100'>
                                    <div className='col-2'>
                                        <p className=' align-middle h-100' style={{border:"0", margin:"0"}}>Features:</p>
                                    </div>
                                    <div className='col-3'>
                                        <select id='bar-selector' className='form-select' onChange={(event) => { setSelectorBar1(event.target.value); handleBarTransformation(data) }}>
                                            {   feature_list.map((feature) => { return ( <option key={feature.name + "bar1"} value={feature.name} selected={"total_eve_calls" === feature.name ? true : false}>{feature.name}</option> ) })  }
                                        </select>
                                    </div>
                                    <div className='col-3'>
                                        <select id='bar-selector' className='form-select' onChange={(event) => { setSelectorBar2(event.target.value); handleBarTransformation(data) }}>
                                            {   feature_list.map((feature) => { return ( <option key={feature.name + "bar1"} value={feature.name} selected={"total_day_calls" === feature.name ? true : false}>{feature.name}</option> ) })  }
                                        </select>
                                    </div>
                                </div>
                                <div className='card-body w-100'>
                                    <HighchartsReact highcharts={Highcharts} options={optionsBar} />
                                </div>
                            </div>
                        </div>
                    </div>
                    <div className='row'>
                        <div className='card graph'>
                            <div className='d-flex flex-row mb-3 mt-5 card-title w-100'>
                                <div className='col-1'>
                                    <p className=' align-middle h-100' style={{border:"0", margin:"0"}}>Features:</p>
                                </div>
                                <div className='col-3'>
                                    <select id='area-selector' className='form-select' onChange={(event) => { setSelectorArea1(event.target.value); handleAreaTransformation(data) } }>
                                        {   feature_list.map((feature) => { return ( <option value={feature.name} selected={"churn" === feature.name ? true : false}>{feature.name}</option> ) })  }
                                    </select>
                                </div>
                                <div className='col-3'>
                                    <select id='area-selector' className='form-select' onChange={(event) => { setSelectorArea2(event.target.value); handleAreaTransformation(data) } }>
                                        {   feature_list.map((feature) => { return ( <option value={feature.name}  selected={"total_day_calls" === feature.name ? true : false}>{feature.name}</option> ) })  }
                                    </select>
                                </div>
                            </div>
                            <div className='card-body w-100'>
                                <HighchartsReact highcharts={Highcharts} options={optionsArea} />
                            </div>
                        </div>
                    </div>
                </div>
            </main>
        </>
    )
}

export default MultipleCustomers