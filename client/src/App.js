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
    let [series, setSeries] = useState([{}])

    const handleScatter = () => {
        var rowValues = document.getElementById("pasteData").value.split("\t"); // Get the pasted row values and split them by tab

        var inputFields = document.querySelectorAll("[type=number]");
        for (var i = 0; i < inputFields.length; i++) {
            inputFields[i].value = rowValues[i];
        }
    }
    const handleClear = () => {
        feature_list.forEach((feature) => {
            document.querySelector("#"+ feature.name).value = ""
        })
    }
    const handleChurn = () => {
        let feature_post = []
        let inputFields = document.querySelectorAll("[type=number]");
        for(let i = 0; i < inputFields.length; i++) {
            let name = inputFields[i].id.split(" ")[0]
            let value = inputFields[i].value
            feature_post.push({
                "name": name,
                "value": value
            })
        }
        console.log("feature_post: ", feature_post)

        fetch("http://localhost:8080/predict", {
            method: "POST",
            mode: "cors",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ feature_post })
        }).then(
            res => res.json()
        ).then(
            data => {
                console.log(data)
                setSeries(data)
            }
        ).catch(
            e => console.error("No prediction yet\n", e)
        )
        
        // fetch("http://localhost:8080/")
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
            <div className='nav'>
                <p className='nav-heading' children={"Churn Prediction"}/>
                <p className='nav-subheading'children={"using Random Forest Classifier"}/>
            </div>
            <div className='container'>
                <Header1 className={"text-center label1"}  children={"Input Your Values"} />
                <div className='d-flex flex-row align-items-center'>
                    <div className='form-floating w-1100'>
                        <Input id={"pasteData"}  className={"form-control d-flex input-row"} type="text" placeholder="Paste Excel row here" />
                        <Label forLabel={"pasteData"} children={"Paste Excel row here"} />
                    </div>
                    <Button className={"d-flex inline scatter-btn"} children={"Scatter Values"} onClick={handleScatter} />
                </div>
                <Header2 className={"mt-3"} children={"...Or input below each value individually"}></Header2>
                <div id={"predictionForm"} className={"form row"}>
                    {/* <Input id={"state"} className={"form-control"} type={"text"} placeholder={"state"} required={"required"} />
                    <Label forLabel={"state"} children={"State"} /> */}
                    <Button theme={"primary"} className={"my-3"} children={"Predict Churn"} onClick={handleChurn} />
                    {feature_list.map((feature) => {
                        return <div className='form-floating col-3'>
                            <InputFeature id={feature.name } className={"form-control d-flex"} key={feature.name} feature={feature.name} />
                            <Label forLabel={feature.name } children={feature.name} />
                        </div>
                    })}
                </div>
                <Button theme={"outline-danger"} className={"my-3"} children={"Clear"} onClick={handleClear}></Button>
            </div>
            <div className='container'>
                <HighchartsReact highcharts={Highcharts} options={options} />
            </div>
        </main>
    )
}

export default App
