import Button from './Components/Button'
import Header1 from './Components/Header1'
import Header2 from './Components/Header2'
import Input from './Components/Input'
import Form from './Components/Form'
import Label from "./Components/Label"
import InputFeature from './Components/InputFeature'
import { useState, useEffect } from 'react'
import Highcharts from 'highcharts'
import HighchartsReact from 'highcharts-react-official'
import './CSS/main.css'

function App() {

    const [feature_list, setData] = useState([{}])
    useEffect(() => {
        fetch("/feature_names").then(
            res => res.json()
        ).then(
            data => {
                setData(data)
            }
        )
    }, [])
    let [series, setSeries] = useState([{}])
    useEffect(() => {
        //fetch("/predict")
    })

    const handleScatter = () => {
        var rowValues = document.getElementById("pasteData").value.split("\t"); // Get the pasted row values and split them by tab

        var inputFields = document.querySelectorAll("#predictionForm")[0];
        for (var i = 0; i < inputFields.length; i++) {
            inputFields[i].value = rowValues[i];
        }
    }
    const handleChurn = () => {
        fetch("http://localhost:8080/predict", {
            method: "POST",
            mode: "cors",
            body: JSON.stringify()
        })
    }
    const handleClear = () => {
        feature_list.forEach((feature) => {
            document.querySelector("#"+ feature.name +"\\ predictionValue").value = ""
        })
    }

    const options = {
        series: series,
        title: {
            text: "Churn of Customer"
        },
        chart: {
            type: "line"
        },
        xAxis: {
            categories: []
        }
    }
    
    return (
        <main className='main'>
            <div className='container'>
                <Header1 className={"display-1"} children={"Customer Churn Analysis"} />
                <Header2 className={"mt-5"} children={"Paste Excel row here."}></Header2>
                <div className='d-flex flex-row align-items-center'>
                    <div className='form-floating w-75'>
                        <Input id={"pasteData"}  className={"w-75 form-control d-flex"} type="text" placeholder="Paste Excel row here" />
                        <Label forLabel={"pasteData"} children={"Paste Excel row here"} />
                    </div>
                    <Button theme={"primary"} className={"d-flex ms-3 h-50 inline"} children={"Scatter Values"} onClick={handleScatter} />
                </div>
                <Header2 className={"mt-3"} children={"...Or input below each value individually"}></Header2>
                <Form id={"predictionForm"} className={"form row"} action={"prediction"} method={"post"}>
                    {/* <Input id={"state"} className={"form-control"} type={"text"} placeholder={"state"} required={"required"} />
                    <Label forLabel={"state"} children={"State"} /> */}
                    <Button theme={"primary"} className={"my-3"} children={"Predict Churn"} onClick={handleChurn} />
                    {feature_list.map((feature) => {
                        return <div className='form-floating col-3'>
                            <InputFeature id={feature.name + " predictionValue"} className={"form-control d-flex"} key={feature.name} feature={feature.name} />
                            <Label forLabel={feature.name + " predictionValue"} children={feature.name} />
                        </div>
                    })}
                </Form>
                <Button theme={"outline-danger"} className={"my-3"} children={"Clear"} onClick={handleClear}></Button>
            </div>
            <div className='container'>
                <HighchartsReact highcharts={Highcharts} options={options} />
            </div>
        </main>
    )
}

export default App
